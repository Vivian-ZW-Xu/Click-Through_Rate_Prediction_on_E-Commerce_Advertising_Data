from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import random
import re
import time
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from torch import nn


INPUT_DIR = Path("data/processed/model_input_f3")
BASE_MANIFEST_PATH = Path("artifacts/encoders/encoding_manifest.json")
F3_MANIFEST_PATH = Path("results/model_input_f3/feature_manifest.json")

DEFAULT_BATCH_SIZE = 8_192
DEFAULT_EPOCHS = 4
DEFAULT_PATIENCE = 1
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-6
DEFAULT_EMBEDDING_DIM = 16
DEFAULT_HIDDEN_DIMS = (256, 128)
DEFAULT_DROPOUT = 0.2
DEFAULT_SEED = 42

SMOKE_TRAIN_ROWS = 300_000
SMOKE_EVAL_ROWS = 150_000
SMOKE_BATCH_SIZE = 4_096
SMOKE_EPOCHS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Wide & Deep or DeepFM with shared F1+F2+F3 inputs."
    )
    parser.add_argument(
        "--model",
        choices=("wide_deep", "deepfm"),
        default="wide_deep",
    )
    parser.add_argument("--run-name", default="wide_deep_f1_f2_f3")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Read the frozen test split only after model selection.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=list(DEFAULT_HIDDEN_DIMS),
    )
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        parser.error("--run-name contains unsupported characters")
    if args.epochs < 1 or args.patience < 0:
        parser.error("--epochs must be positive and --patience non-negative")
    if args.batch_size < 256:
        parser.error("--batch-size must be at least 256")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("invalid optimizer settings")
    if args.embedding_dim < 2:
        parser.error("--embedding-dim must be at least 2")
    if not args.hidden_dims or min(args.hidden_dims) < 1:
        parser.error("--hidden-dims must contain positive integers")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    return args


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def feature_schema() -> tuple[list[str], list[str], list[str], dict[str, int]]:
    base = load_json(BASE_MANIFEST_PATH)
    f3 = load_json(F3_MANIFEST_PATH)
    categorical = [
        feature
        for feature in base["categorical_features"]
        if feature != "day_of_week_idx"
    ]
    binary = [*base["binary_features"], *f3["binary_features"]]
    numeric = [*base["numeric_features"], *f3["numeric_features"]]
    vocabulary_sizes = {
        feature: int(base["vocabulary_sizes"][feature])
        for feature in categorical
    }
    all_features = [*categorical, *binary, *numeric]
    if len(all_features) != len(set(all_features)):
        raise ValueError("Duplicate model features found")
    return categorical, binary, numeric, vocabulary_sizes


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def split_path(split: str) -> Path:
    path = INPUT_DIR / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing model input: {path}")
    return path


def validate_schema(path: Path, columns: list[str]) -> None:
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(columns) - schema_names)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def arrow_column_to_numpy(batch: pa.RecordBatch, name: str, dtype: np.dtype) -> np.ndarray:
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise KeyError(name)
    array = batch.column(index)
    if array.null_count:
        raise ValueError(f"Null values found in model column {name}")
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=dtype)


def iter_record_batches(
    path: Path,
    columns: list[str],
    batch_size: int,
    row_limit: int | None,
) -> Iterator[pa.RecordBatch]:
    seen = 0
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=columns,
        use_threads=True,
    ):
        if row_limit is not None:
            remaining = row_limit - seen
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
        if batch.num_rows:
            yield batch
            seen += batch.num_rows
        if row_limit is not None and seen >= row_limit:
            break


def dense_numpy(
    batch: pa.RecordBatch,
    binary_features: list[str],
    numeric_features: list[str],
    numeric_mean: np.ndarray,
    numeric_scale: np.ndarray,
) -> np.ndarray:
    binary = np.column_stack(
        [arrow_column_to_numpy(batch, name, np.float32) for name in binary_features]
    )
    numeric = np.column_stack(
        [arrow_column_to_numpy(batch, name, np.float32) for name in numeric_features]
    )
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite numeric feature found")
    numeric = (numeric - numeric_mean) / numeric_scale
    return np.ascontiguousarray(
        np.concatenate([binary, numeric], axis=1),
        dtype=np.float32,
    )


def categorical_numpy(batch: pa.RecordBatch, features: list[str]) -> np.ndarray:
    return np.ascontiguousarray(
        np.column_stack(
            [arrow_column_to_numpy(batch, name, np.int64) for name in features]
        ),
        dtype=np.int64,
    )


def fit_numeric_scaler(
    path: Path,
    numeric_features: list[str],
    batch_size: int,
    row_limit: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    count = 0
    feature_sum = np.zeros(len(numeric_features), dtype=np.float64)
    feature_square_sum = np.zeros(len(numeric_features), dtype=np.float64)
    for batch in iter_record_batches(path, numeric_features, batch_size, row_limit):
        values = np.column_stack(
            [arrow_column_to_numpy(batch, name, np.float64) for name in numeric_features]
        )
        if not np.isfinite(values).all():
            raise ValueError("Non-finite values found while fitting scaler")
        feature_sum += values.sum(axis=0)
        feature_square_sum += np.square(values).sum(axis=0)
        count += values.shape[0]
    if count == 0:
        raise ValueError("No rows available for numeric scaler")
    mean = feature_sum / count
    variance = np.maximum(feature_square_sum / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32), count


class TabularCTRModel(nn.Module):
    def __init__(
        self,
        model_type: str,
        vocabulary_sizes: list[int],
        dense_dimension: int,
        embedding_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim) for size in vocabulary_sizes]
        )
        self.first_order = nn.ModuleList(
            [nn.Embedding(size, 1) for size in vocabulary_sizes]
        )
        self.wide_dense = nn.Linear(dense_dimension, 1)

        deep_input = len(vocabulary_sizes) * embedding_dim + dense_dimension
        layers: list[nn.Module] = []
        previous = deep_input
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.deep = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
        for embedding in self.first_order:
            nn.init.zeros_(embedding.weight)
        nn.init.zeros_(self.wide_dense.bias)

    def forward(self, categorical: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        embedded = torch.stack(
            [
                embedding(categorical[:, index])
                for index, embedding in enumerate(self.embeddings)
            ],
            dim=1,
        )
        first_order = torch.stack(
            [
                embedding(categorical[:, index]).squeeze(-1)
                for index, embedding in enumerate(self.first_order)
            ],
            dim=1,
        ).sum(dim=1, keepdim=True)
        wide_logit = first_order + self.wide_dense(dense)
        deep_input = torch.cat([embedded.flatten(start_dim=1), dense], dim=1)
        logit = wide_logit + self.deep(deep_input)

        if self.model_type == "deepfm":
            summed = embedded.sum(dim=1)
            fm_logit = 0.5 * (
                summed.square() - embedded.square().sum(dim=1)
            ).sum(dim=1, keepdim=True)
            logit = logit + fm_logit
        return logit.squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def validate_category_ranges(
    categorical: np.ndarray,
    vocabulary_sizes: list[int],
    feature_names: list[str],
) -> None:
    for index, (size, name) in enumerate(zip(vocabulary_sizes, feature_names)):
        values = categorical[:, index]
        if values.min(initial=0) < 0 or values.max(initial=0) >= size:
            raise ValueError(
                f"Categorical index outside vocabulary for {name}: "
                f"min={values.min()}, max={values.max()}, size={size}"
            )


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    path: Path,
    categorical_features: list[str],
    binary_features: list[str],
    numeric_features: list[str],
    vocabulary_sizes: list[int],
    numeric_mean: np.ndarray,
    numeric_scale: np.ndarray,
    batch_size: int,
    row_limit: int | None,
    device: torch.device,
    seed: int,
) -> tuple[int, float]:
    model.train()
    columns = [*categorical_features, *binary_features, *numeric_features, "clk"]
    rows = 0
    loss_sum = 0.0
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    for batch in iter_record_batches(path, columns, batch_size, row_limit):
        categorical = categorical_numpy(batch, categorical_features)
        dense = dense_numpy(
            batch,
            binary_features,
            numeric_features,
            numeric_mean,
            numeric_scale,
        )
        labels = arrow_column_to_numpy(batch, "clk", np.float32)
        order = rng.permutation(labels.size)
        categorical = categorical[order]
        dense = dense[order]
        labels = labels[order]
        validate_category_ranges(categorical, vocabulary_sizes, categorical_features)

        categorical_tensor = torch.from_numpy(categorical).to(device)
        dense_tensor = torch.from_numpy(dense).to(device)
        label_tensor = torch.from_numpy(labels).to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(categorical_tensor, dense_tensor)
        loss = loss_function(logits, label_tensor)
        loss.backward()
        optimizer.step()

        batch_rows = labels.size
        rows += batch_rows
        loss_sum += float(loss.detach().cpu()) * batch_rows
        if rows % max(batch_size * 250, batch_size) < batch_size:
            elapsed = time.perf_counter() - started
            print(f"  rows={rows:,} rate={rows / max(elapsed, 1e-9):,.0f} rows/s")

        del categorical_tensor, dense_tensor, label_tensor, logits, loss

    return rows, loss_sum / rows


@torch.no_grad()
def predict_split(
    model: nn.Module,
    split: str,
    categorical_features: list[str],
    binary_features: list[str],
    numeric_features: list[str],
    vocabulary_sizes: list[int],
    numeric_mean: np.ndarray,
    numeric_scale: np.ndarray,
    batch_size: int,
    row_limit: int | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    columns = [
        *categorical_features,
        *binary_features,
        *numeric_features,
        "impression_id",
        "clk",
    ]
    impression_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    for batch in iter_record_batches(split_path(split), columns, batch_size, row_limit):
        categorical = categorical_numpy(batch, categorical_features)
        dense = dense_numpy(
            batch,
            binary_features,
            numeric_features,
            numeric_mean,
            numeric_scale,
        )
        validate_category_ranges(categorical, vocabulary_sizes, categorical_features)
        logits = model(
            torch.from_numpy(categorical).to(device),
            torch.from_numpy(dense).to(device),
        )
        predictions = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        impression_parts.append(
            arrow_column_to_numpy(batch, "impression_id", np.uint32)
        )
        label_parts.append(arrow_column_to_numpy(batch, "clk", np.uint8))
        prediction_parts.append(predictions)
    return (
        np.concatenate(impression_parts),
        np.concatenate(label_parts),
        np.concatenate(prediction_parts),
    )


def metric_row(
    run_name: str,
    split: str,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    clipped = np.clip(predictions.astype(np.float64), 1e-7, 1 - 1e-7)
    return {
        "model": run_name,
        "split": split,
        "rows": int(labels.size),
        "clicks": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "mean_prediction": float(predictions.mean()),
        "auc": float(roc_auc_score(labels, predictions)),
        "log_loss": float(log_loss(labels, clipped)),
        "pr_auc": float(average_precision_score(labels, predictions)),
        "brier_score": float(brier_score_loss(labels, predictions)),
    }


def save_predictions(
    path: Path,
    impression_id: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "impression_id": impression_id,
            "clk": labels,
            "prediction": predictions,
        }
    ).write_parquet(path, compression="zstd", statistics=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    run_name = f"{args.run_name}_smoke" if args.smoke_test else args.run_name
    epochs = SMOKE_EPOCHS if args.smoke_test else args.epochs
    batch_size = SMOKE_BATCH_SIZE if args.smoke_test else args.batch_size
    train_limit = SMOKE_TRAIN_ROWS if args.smoke_test else None
    eval_limit = SMOKE_EVAL_ROWS if args.smoke_test else None

    categorical, binary, numeric, vocabulary_map = feature_schema()
    vocabulary_sizes = [vocabulary_map[name] for name in categorical]
    required = [*categorical, *binary, *numeric, "impression_id", "clk"]
    for split in ("train", "val", "test"):
        validate_schema(split_path(split), required)

    result_dir = Path("results") / run_name
    model_dir = Path("artifacts/models") / run_name
    prediction_dir = Path("artifacts/predictions") / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TABULAR NEURAL CTR TRAINING")
    print("=" * 80)
    print(f"Run:                  {run_name}")
    print(f"Model:                {args.model}")
    print(f"Device:               {device}")
    print(f"Categorical features: {len(categorical)}")
    print(f"Binary features:      {len(binary)}")
    print(f"Numeric features:     {len(numeric)}")
    print(f"Embedding dimension:  {args.embedding_dim}")
    print(f"Hidden dimensions:    {args.hidden_dims}")
    print(f"Batch size:           {batch_size:,}")
    print(f"Epochs:               {epochs}")
    print(f"Evaluate frozen test: {args.evaluate_test}")

    print("\nFitting train-only numeric scaler...")
    numeric_mean, numeric_scale, scaler_rows = fit_numeric_scaler(
        split_path("train"),
        numeric,
        max(batch_size, 65_536),
        train_limit,
    )
    scaler_payload = {
        "features": numeric,
        "mean": numeric_mean.tolist(),
        "scale": numeric_scale.tolist(),
        "rows": scaler_rows,
        "train_only": True,
    }
    (result_dir / "numeric_scaler.json").write_text(
        json.dumps(scaler_payload, indent=2), encoding="utf-8"
    )

    model = TabularCTRModel(
        model_type=args.model,
        vocabulary_sizes=vocabulary_sizes,
        dense_dimension=len(binary) + len(numeric),
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()
    print(f"Trainable parameters: {parameter_count(model):,}")

    checkpoint_path = model_dir / "best_model.pt"
    history: list[dict[str, object]] = []
    best_log_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    total_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        print("\n" + "-" * 80)
        print(f"EPOCH {epoch}/{epochs}")
        print("-" * 80)
        epoch_started = time.perf_counter()
        train_rows, train_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            path=split_path("train"),
            categorical_features=categorical,
            binary_features=binary,
            numeric_features=numeric,
            vocabulary_sizes=vocabulary_sizes,
            numeric_mean=numeric_mean,
            numeric_scale=numeric_scale,
            batch_size=batch_size,
            row_limit=train_limit,
            device=device,
            seed=args.seed + epoch,
        )
        _, val_labels, val_predictions = predict_split(
            model,
            "val",
            categorical,
            binary,
            numeric,
            vocabulary_sizes,
            numeric_mean,
            numeric_scale,
            batch_size,
            eval_limit,
            device,
        )
        val_metric = metric_row(run_name, "val", val_labels, val_predictions)
        epoch_minutes = (time.perf_counter() - epoch_started) / 60
        print(
            f"Train loss={train_loss:.6f}; val AUC={val_metric['auc']:.6f}; "
            f"val LogLoss={val_metric['log_loss']:.6f}; "
            f"mean(p)={val_metric['mean_prediction']:.6f}; "
            f"time={epoch_minutes:.1f}m"
        )
        history.append(
            {
                "epoch": epoch,
                "train_rows": train_rows,
                "train_loss": train_loss,
                "epoch_minutes": epoch_minutes,
                "val_auc": val_metric["auc"],
                "val_log_loss": val_metric["log_loss"],
                "val_pr_auc": val_metric["pr_auc"],
                "val_mean_prediction": val_metric["mean_prediction"],
            }
        )
        if float(val_metric["log_loss"]) < best_log_loss - 1e-7:
            best_log_loss = float(val_metric["log_loss"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model,
                    "categorical_features": categorical,
                    "binary_features": binary,
                    "numeric_features": numeric,
                    "vocabulary_sizes": vocabulary_map,
                    "embedding_dim": args.embedding_dim,
                    "hidden_dims": args.hidden_dims,
                    "dropout": args.dropout,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
            print(f"New best checkpoint saved at epoch {epoch}.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement; best epoch remains {best_epoch}.")
            if epochs_without_improvement > args.patience:
                print("Early stopping triggered.")
                break
        del val_labels, val_predictions
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("\n" + "=" * 80)
    print(f"FINAL EVALUATION USING BEST EPOCH {best_epoch}")
    print("=" * 80)

    metric_rows: list[dict[str, object]] = []
    evaluation_splits = ["val", *( ["test"] if args.evaluate_test else [] )]
    for split in evaluation_splits:
        impression_id, labels, predictions = predict_split(
            model,
            split,
            categorical,
            binary,
            numeric,
            vocabulary_sizes,
            numeric_mean,
            numeric_scale,
            batch_size,
            eval_limit,
            device,
        )
        metrics = metric_row(run_name, split, labels, predictions)
        metric_rows.append(metrics)
        print(
            f"  {split:<5} rows={metrics['rows']:,} AUC={metrics['auc']:.6f} "
            f"LogLoss={metrics['log_loss']:.6f} PR-AUC={metrics['pr_auc']:.6f} "
            f"mean(p)={metrics['mean_prediction']:.6f}"
        )
        save_predictions(
            prediction_dir / f"{split}_predictions.parquet",
            impression_id,
            labels,
            predictions,
        )

    pl.DataFrame(history).write_csv(result_dir / "training_history.csv")
    pl.DataFrame(metric_rows).write_csv(result_dir / "metrics.csv")
    run_manifest = {
        "run_name": run_name,
        "model": args.model,
        "feature_set": "f1_f2_f3",
        "input_directory": str(INPUT_DIR),
        "device": str(device),
        "categorical_features": categorical,
        "binary_features": binary,
        "numeric_features": numeric,
        "vocabulary_sizes": vocabulary_map,
        "embedding_dim": args.embedding_dim,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "class_weighting": "none",
        "evaluate_test": args.evaluate_test,
        "smoke_test": args.smoke_test,
        "train_limit": train_limit,
        "eval_limit": eval_limit,
        "parameter_count": parameter_count(model),
        "training_minutes": (time.perf_counter() - total_started) / 60,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "polars": pl.__version__,
            "pyarrow": pa.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    (result_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("TABULAR NEURAL TRAINING COMPLETED")
    print("=" * 80)
    print(f"Best model:  {checkpoint_path}")
    print(f"Metrics:     {result_dir / 'metrics.csv'}")
    print(f"Predictions: {prediction_dir}")
    if args.smoke_test:
        print("\nSmoke test passed. Do not use smoke metrics as final evidence.")


if __name__ == "__main__":
    main()
