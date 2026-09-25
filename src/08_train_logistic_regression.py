"""Train a leakage-free, out-of-core logistic-regression CTR baseline.

The encoded CTR tables contain more than 20 million training rows and several
million possible one-hot categories.  Building the full design matrix in RAM
would be wasteful, so this script:

1. Fits numeric scaling parameters on the training split only.
2. Reads Parquet row groups in batches.
3. Builds a temporary sparse one-hot matrix for each batch.
4. Optimizes logistic loss with SGDClassifier.partial_fit.
5. Selects the best epoch using validation Log Loss.
6. Evaluates the frozen model once on the test split.

No label-derived feature is created here.  Validation and test data are never
used to fit category vocabularies, scalers, or model parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import time
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


INPUT_DIRS = {
    "f1": Path("data/processed/model_input"),
    "f1_f2": Path("data/processed/model_input"),
    "f1_f2_f3": Path("data/processed/model_input_f3"),
}
INPUT_DIR = INPUT_DIRS["f1_f2"]
ENCODING_MANIFEST_PATH = Path(
    "artifacts/encoders/encoding_manifest.json"
)
F3_FEATURE_MANIFEST_PATH = Path(
    "results/model_input_f3/feature_manifest.json"
)
F1_BINARY_FEATURES = [
    "price_is_sentinel",
    "brand_is_missing",
    "has_profile",
]
F1_NUMERIC_FEATURES = ["log_price"]
PROTOCOL_EXCLUDED_FEATURES = {"day_of_week_idx"}

SPLITS = ("train", "val", "test")
EXPECTED_ROWS = {
    "train": 20_015_245,
    "val": 3_234_051,
    "test": 3_308_665,
}

LABEL_COLUMN = "clk"
ID_COLUMN = "impression_id"

DEFAULT_BATCH_SIZE = 200_000
DEFAULT_EPOCHS = 2
DEFAULT_ALPHA = 1e-6
RANDOM_SEED = 42

SMOKE_TRAIN_ROWS = 500_000
SMOKE_EVAL_ROWS = 200_000
SMOKE_BATCH_SIZE = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an out-of-core sparse logistic-regression CTR baseline."
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of passes over the complete training split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Maximum rows per temporary sparse matrix.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="L2 regularization strength used by SGDClassifier.",
    )
    parser.add_argument(
        "--eta0",
        type=float,
        default=0.005,
        help="Initial learning rate for constant-rate SGD (default: 0.005).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for row-group and within-batch shuffling.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a small end-to-end check without overwriting the full run. "
            f"Uses {SMOKE_TRAIN_ROWS:,} train rows and "
            f"{SMOKE_EVAL_ROWS:,} rows per evaluation split."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="logistic_regression",
        help=(
            "Output name under results/, artifacts/models/, and "
            "artifacts/predictions/."
        ),
    )
    parser.add_argument(
        "--feature-set",
        choices=("f1", "f1_f2", "f1_f2_f3"),
        default="f1_f2",
        help=(
            "Observable-information budget. f1 uses static/context features; "
            "f1_f2 adds aggregate behavior; f1_f2_f3 additionally uses "
            "past-only CTR statistics. day_of_week_idx is excluded from all "
            "three by the fixed experimental protocol."
        ),
    )
    parser.add_argument(
        "--exclude-features",
        nargs="*",
        default=[],
        help=(
            "Encoded feature columns to exclude from this run. "
            "Useful for controlled feature ablations."
        ),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Evaluate and save predictions for the frozen test split. "
            "Leave disabled during hyperparameter selection."
        ),
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 10_000:
        parser.error("--batch-size must be at least 10,000")
    if args.alpha <= 0:
        parser.error("--alpha must be positive")
    if args.eta0 <= 0:
        parser.error("--eta0 must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        parser.error(
            "--run-name may contain only letters, numbers, dot, "
            "underscore, and hyphen"
        )

    return args


def input_path(split_name: str) -> Path:
    return INPUT_DIR / f"{split_name}.parquet"


def load_encoding_manifest() -> dict:
    if not ENCODING_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing encoding manifest: {ENCODING_MANIFEST_PATH}\n"
            "Run src/07_build_model_inputs.py first."
        )

    with ENCODING_MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    required_keys = {
        "categorical_features",
        "vocabulary_sizes",
        "binary_features",
        "numeric_features",
        "label",
    }
    missing = required_keys - set(manifest)
    if missing:
        raise ValueError(
            f"Encoding manifest is missing keys: {sorted(missing)}"
        )

    if manifest["label"] != LABEL_COLUMN:
        raise ValueError(
            f"Expected label {LABEL_COLUMN!r}, found {manifest['label']!r}"
        )

    return manifest


def validate_inputs(required_columns: list[str], smoke_test: bool) -> None:
    for split_name in SPLITS:
        path = input_path(split_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model-input table: {path}\n"
                "Run src/07_build_model_inputs.py first."
            )

        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        missing = set(required_columns) - available
        if missing:
            raise ValueError(
                f"{split_name}: required columns are missing: "
                f"{sorted(missing)}"
            )

        rows = parquet_file.metadata.num_rows
        if not smoke_test and rows != EXPECTED_ROWS[split_name]:
            raise ValueError(
                f"{split_name}: expected {EXPECTED_ROWS[split_name]:,} "
                f"rows, found {rows:,}"
            )


def parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def batch_column(
    batch: pa.RecordBatch,
    name: str,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Column {name!r} is absent from a record batch")
    values = batch.column(index).to_numpy(zero_copy_only=False)
    if dtype is not None:
        values = values.astype(dtype, copy=False)
    return values


def iterate_parquet_batches(
    path: Path,
    columns: list[str],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    max_rows: int | None,
) -> Iterator[tuple[pa.RecordBatch, np.ndarray | None]]:
    """Yield bounded record batches, optionally shuffling row groups and rows."""

    parquet_file = pq.ParquetFile(path)
    row_groups = np.arange(
        parquet_file.num_row_groups,
        dtype=np.int32,
    )
    rng = np.random.default_rng(seed)

    if shuffle:
        rng.shuffle(row_groups)

    rows_yielded = 0

    for row_group in row_groups:
        iterator = parquet_file.iter_batches(
            batch_size=batch_size,
            row_groups=[int(row_group)],
            columns=columns,
            use_threads=True,
        )

        for batch in iterator:
            if max_rows is not None:
                remaining = max_rows - rows_yielded
                if remaining <= 0:
                    return
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)

            permutation = None
            if shuffle and batch.num_rows > 1:
                permutation = rng.permutation(batch.num_rows)

            rows_yielded += batch.num_rows
            yield batch, permutation

            if max_rows is not None and rows_yielded >= max_rows:
                return


def apply_permutation(
    values: np.ndarray,
    permutation: np.ndarray | None,
) -> np.ndarray:
    if permutation is None:
        return values
    return values[permutation]


def extract_dense_values(
    batch: pa.RecordBatch,
    features: list[str],
    permutation: np.ndarray | None,
) -> np.ndarray:
    values = np.column_stack(
        [
            batch_column(batch, feature, np.float32)
            for feature in features
        ]
    )
    return apply_permutation(values, permutation)


def fit_numeric_scaler(
    train_path: Path,
    numeric_features: list[str],
    batch_size: int,
    max_rows: int | None,
) -> StandardScaler:
    print("\nFitting numeric scaler on training data only...")
    started = time.perf_counter()
    scaler = StandardScaler()
    rows_seen = 0

    for batch, _ in iterate_parquet_batches(
        train_path,
        numeric_features,
        batch_size,
        shuffle=False,
        seed=RANDOM_SEED,
        max_rows=max_rows,
    ):
        numeric = extract_dense_values(
            batch,
            numeric_features,
            permutation=None,
        )
        scaler.partial_fit(numeric)
        rows_seen += batch.num_rows

        if rows_seen % 2_000_000 < batch.num_rows:
            print(f"  scaler rows: {rows_seen:,}")

    elapsed = time.perf_counter() - started
    print(
        f"Numeric scaler fitted on {rows_seen:,} rows "
        f"in {elapsed / 60:.1f} minutes."
    )
    return scaler


def build_feature_layout(
    categorical_features: list[str],
    vocabulary_sizes: dict[str, int],
    binary_features: list[str],
    numeric_features: list[str],
) -> dict:
    offsets: dict[str, int] = {}
    cursor = 0

    for feature in categorical_features:
        if feature not in vocabulary_sizes:
            raise ValueError(
                f"Missing vocabulary size for categorical feature {feature}"
            )
        size = int(vocabulary_sizes[feature])
        if size < 3:
            raise ValueError(
                f"Invalid vocabulary size for {feature}: {size}"
            )
        offsets[feature] = cursor
        cursor += size

    dense_features = [*binary_features, *numeric_features]
    dense_offsets = {
        feature: cursor + index
        for index, feature in enumerate(dense_features)
    }

    return {
        "categorical_offsets": offsets,
        "categorical_dimension": cursor,
        "dense_offsets": dense_offsets,
        "dense_features": dense_features,
        "total_dimension": cursor + len(dense_features),
    }


def build_sparse_matrix(
    batch: pa.RecordBatch,
    permutation: np.ndarray | None,
    categorical_features: list[str],
    binary_features: list[str],
    numeric_features: list[str],
    vocabulary_sizes: dict[str, int],
    feature_layout: dict,
    scaler: StandardScaler,
) -> sp.csr_matrix:
    n_rows = batch.num_rows
    n_categorical = len(categorical_features)

    category_values = np.column_stack(
        [
            batch_column(batch, feature, np.int64)
            for feature in categorical_features
        ]
    )
    category_values = apply_permutation(
        category_values,
        permutation,
    )

    for index, feature in enumerate(categorical_features):
        values = category_values[:, index]
        minimum = int(values.min())
        maximum = int(values.max())
        vocabulary_size = int(vocabulary_sizes[feature])

        # PAD=0 is reserved for future sequences and must not occur in a
        # tabular impression. UNKNOWN=1 is valid.
        if minimum < 1 or maximum >= vocabulary_size:
            raise ValueError(
                f"{feature}: encoded range [{minimum}, {maximum}] is "
                f"outside [1, {vocabulary_size - 1}]"
            )

    offsets = np.asarray(
        [
            feature_layout["categorical_offsets"][feature]
            for feature in categorical_features
        ],
        dtype=np.int64,
    )
    category_columns = (
        category_values + offsets[None, :]
    ).ravel(order="C").astype(np.int32, copy=False)
    category_rows = np.repeat(
        np.arange(n_rows, dtype=np.int32),
        n_categorical,
    )
    category_data = np.ones(
        n_rows * n_categorical,
        dtype=np.float32,
    )

    categorical_matrix = sp.csr_matrix(
        (category_data, (category_rows, category_columns)),
        shape=(
            n_rows,
            feature_layout["categorical_dimension"],
        ),
        dtype=np.float32,
    )

    binary_values = extract_dense_values(
        batch,
        binary_features,
        permutation,
    )
    numeric_values = extract_dense_values(
        batch,
        numeric_features,
        permutation,
    )
    numeric_values = scaler.transform(numeric_values).astype(
        np.float32,
        copy=False,
    )

    dense_values = np.column_stack(
        [binary_values, numeric_values]
    ).astype(np.float32, copy=False)
    dense_matrix = sp.csr_matrix(dense_values, dtype=np.float32)

    design_matrix = sp.hstack(
        [categorical_matrix, dense_matrix],
        format="csr",
        dtype=np.float32,
    )
    design_matrix.sort_indices()

    if design_matrix.shape[1] != feature_layout["total_dimension"]:
        raise ValueError(
            "Sparse design matrix has an unexpected feature dimension"
        )

    return design_matrix


def metrics_from_arrays(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
    return {
        "rows": int(labels.size),
        "clicks": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "mean_prediction": float(predictions.mean()),
        "auc": float(roc_auc_score(labels, predictions)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "pr_auc": float(average_precision_score(labels, predictions)),
        "brier_score": float(brier_score_loss(labels, predictions)),
    }


def evaluate_split(
    model: SGDClassifier,
    split_name: str,
    columns: list[str],
    categorical_features: list[str],
    binary_features: list[str],
    numeric_features: list[str],
    vocabulary_sizes: dict[str, int],
    feature_layout: dict,
    scaler: StandardScaler,
    batch_size: int,
    max_rows: int | None,
    prediction_path: Path | None,
) -> dict[str, float | int]:
    started = time.perf_counter()
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    writer: pq.ParquetWriter | None = None
    rows_seen = 0

    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for batch, _ in iterate_parquet_batches(
            input_path(split_name),
            columns,
            batch_size,
            shuffle=False,
            seed=RANDOM_SEED,
            max_rows=max_rows,
        ):
            matrix = build_sparse_matrix(
                batch=batch,
                permutation=None,
                categorical_features=categorical_features,
                binary_features=binary_features,
                numeric_features=numeric_features,
                vocabulary_sizes=vocabulary_sizes,
                feature_layout=feature_layout,
                scaler=scaler,
            )
            labels = batch_column(batch, LABEL_COLUMN, np.uint8)
            predictions = model.predict_proba(matrix)[:, 1].astype(
                np.float32,
                copy=False,
            )

            all_labels.append(labels)
            all_predictions.append(predictions)
            rows_seen += batch.num_rows

            if prediction_path is not None:
                impression_ids = batch_column(batch, ID_COLUMN)
                prediction_table = pa.table(
                    {
                        ID_COLUMN: pa.array(impression_ids),
                        LABEL_COLUMN: pa.array(labels),
                        "prediction": pa.array(predictions),
                    }
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        prediction_path,
                        prediction_table.schema,
                        compression="zstd",
                    )
                writer.write_table(prediction_table)

    finally:
        if writer is not None:
            writer.close()

    labels_array = np.concatenate(all_labels)
    predictions_array = np.concatenate(all_predictions)
    metrics = metrics_from_arrays(labels_array, predictions_array)
    elapsed = time.perf_counter() - started

    print(
        f"  {split_name:<5} rows={rows_seen:,} "
        f"AUC={metrics['auc']:.6f} "
        f"LogLoss={metrics['log_loss']:.6f} "
        f"PR-AUC={metrics['pr_auc']:.6f} "
        f"mean(p)={metrics['mean_prediction']:.6f} "
        f"time={elapsed / 60:.1f}m"
    )

    return metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_model(args: argparse.Namespace) -> None:
    global INPUT_DIR
    INPUT_DIR = INPUT_DIRS[args.feature_set]
    manifest = load_encoding_manifest()

    categorical_features = list(manifest["categorical_features"])
    if args.feature_set == "f1":
        binary_features = list(F1_BINARY_FEATURES)
        numeric_features = list(F1_NUMERIC_FEATURES)
    else:
        binary_features = list(manifest["binary_features"])
        numeric_features = list(manifest["numeric_features"])

    if args.feature_set == "f1_f2_f3":
        if not F3_FEATURE_MANIFEST_PATH.exists():
            raise FileNotFoundError(
                f"Missing {F3_FEATURE_MANIFEST_PATH}. "
                "Run src/12_build_f3_model_inputs.py first."
            )
        with F3_FEATURE_MANIFEST_PATH.open("r", encoding="utf-8") as file:
            f3_manifest = json.load(file)
        binary_features.extend(f3_manifest["binary_features"])
        numeric_features.extend(f3_manifest["numeric_features"])

    duplicate_features = {
        feature
        for feature in [*categorical_features, *binary_features, *numeric_features]
        if [*categorical_features, *binary_features, *numeric_features].count(feature) > 1
    }
    if duplicate_features:
        raise ValueError(f"Duplicate model features: {sorted(duplicate_features)}")

    available_features = {
        *categorical_features,
        *binary_features,
        *numeric_features,
    }
    excluded_features = set(args.exclude_features) | PROTOCOL_EXCLUDED_FEATURES
    unknown_exclusions = excluded_features - available_features
    if unknown_exclusions:
        raise ValueError(
            "Cannot exclude unknown model features: "
            f"{sorted(unknown_exclusions)}"
        )

    categorical_features = [
        feature
        for feature in categorical_features
        if feature not in excluded_features
    ]
    binary_features = [
        feature
        for feature in binary_features
        if feature not in excluded_features
    ]
    numeric_features = [
        feature
        for feature in numeric_features
        if feature not in excluded_features
    ]
    vocabulary_sizes = {
        key: int(value)
        for key, value in manifest["vocabulary_sizes"].items()
    }

    model_columns = [
        ID_COLUMN,
        LABEL_COLUMN,
        *categorical_features,
        *binary_features,
        *numeric_features,
    ]

    validate_inputs(model_columns, smoke_test=args.smoke_test)

    run_name = (
        f"{args.run_name}_smoke"
        if args.smoke_test
        else args.run_name
    )
    result_dir = Path("results") / run_name
    model_dir = Path("artifacts/models") / run_name
    prediction_dir = Path("artifacts/predictions") / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        epochs = 1
        batch_size = min(args.batch_size, SMOKE_BATCH_SIZE)
        max_train_rows = SMOKE_TRAIN_ROWS
        max_eval_rows = SMOKE_EVAL_ROWS
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        max_train_rows = None
        max_eval_rows = None

    feature_layout = build_feature_layout(
        categorical_features,
        vocabulary_sizes,
        binary_features,
        numeric_features,
    )

    print("=" * 80)
    print("OUT-OF-CORE LOGISTIC REGRESSION")
    print("=" * 80)
    print(f"Run:                   {run_name}")
    print(f"Feature set:           {args.feature_set}")
    print(f"Input directory:       {INPUT_DIR}")
    print(f"Categorical features:  {len(categorical_features)}")
    print(f"Binary features:       {len(binary_features)}")
    print(f"Numeric features:      {len(numeric_features)}")
    print(
        "Excluded features:     "
        f"{sorted(excluded_features) if excluded_features else 'none'}"
    )
    print(
        f"Sparse dimensions:     "
        f"{feature_layout['total_dimension']:,}"
    )
    print(f"Batch size:            {batch_size:,}")
    print(f"Epochs:                {epochs}")
    print(f"L2 alpha:              {args.alpha:g}")
    print(f"Learning rate eta0:    {args.eta0:g}")
    print("Class weighting:       none (preserves CTR probabilities)")
    print(f"Evaluate frozen test:  {args.evaluate_test}")
    if args.smoke_test:
        print(f"Smoke train rows:      {max_train_rows:,}")
        print(f"Smoke eval rows:       {max_eval_rows:,}")

    scaler = fit_numeric_scaler(
        input_path("train"),
        numeric_features,
        batch_size,
        max_train_rows,
    )

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.alpha,
        fit_intercept=True,
        learning_rate="constant",
        eta0=args.eta0,
        random_state=args.seed,
        average=1_000_000,
        class_weight=None,
    )

    best_model_path = model_dir / "best_model.joblib"
    training_history: list[dict] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    initialized = False
    train_required_columns = [
        LABEL_COLUMN,
        *categorical_features,
        *binary_features,
        *numeric_features,
    ]

    total_train_rows = (
        max_train_rows
        if max_train_rows is not None
        else parquet_rows(input_path("train"))
    )

    for epoch in range(1, epochs + 1):
        print("\n" + "-" * 80)
        print(f"TRAINING EPOCH {epoch}/{epochs}")
        print("-" * 80)
        started = time.perf_counter()
        rows_seen = 0
        next_report = 2_000_000

        for batch, permutation in iterate_parquet_batches(
            input_path("train"),
            train_required_columns,
            batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            max_rows=max_train_rows,
        ):
            matrix = build_sparse_matrix(
                batch=batch,
                permutation=permutation,
                categorical_features=categorical_features,
                binary_features=binary_features,
                numeric_features=numeric_features,
                vocabulary_sizes=vocabulary_sizes,
                feature_layout=feature_layout,
                scaler=scaler,
            )
            labels = apply_permutation(
                batch_column(batch, LABEL_COLUMN, np.uint8),
                permutation,
            )

            if not initialized:
                model.partial_fit(
                    matrix,
                    labels,
                    classes=np.asarray([0, 1], dtype=np.uint8),
                )
                initialized = True
            else:
                model.partial_fit(matrix, labels)

            rows_seen += batch.num_rows
            if rows_seen >= next_report or rows_seen == total_train_rows:
                elapsed = time.perf_counter() - started
                rate = rows_seen / max(elapsed, 1e-9)
                print(
                    f"  rows={rows_seen:,}/{total_train_rows:,} "
                    f"({100 * rows_seen / total_train_rows:.1f}%) "
                    f"rate={rate:,.0f} rows/s"
                )
                next_report += 2_000_000

        train_minutes = (time.perf_counter() - started) / 60
        print(f"Epoch {epoch} training time: {train_minutes:.1f} minutes")

        print("Validation:")
        validation_metrics = evaluate_split(
            model=model,
            split_name="val",
            columns=model_columns,
            categorical_features=categorical_features,
            binary_features=binary_features,
            numeric_features=numeric_features,
            vocabulary_sizes=vocabulary_sizes,
            feature_layout=feature_layout,
            scaler=scaler,
            batch_size=batch_size,
            max_rows=max_eval_rows,
            prediction_path=None,
        )

        history_row = {
            "epoch": epoch,
            "train_rows": rows_seen,
            "train_minutes": train_minutes,
            "val_rows": validation_metrics["rows"],
            "val_auc": validation_metrics["auc"],
            "val_log_loss": validation_metrics["log_loss"],
            "val_pr_auc": validation_metrics["pr_auc"],
            "val_brier_score": validation_metrics["brier_score"],
        }
        training_history.append(history_row)
        write_csv(
            result_dir / "training_history.csv",
            training_history,
        )

        if validation_metrics["log_loss"] < best_validation_loss:
            best_validation_loss = float(validation_metrics["log_loss"])
            best_epoch = epoch
            joblib.dump(
                {
                    "model": model,
                    "scaler": scaler,
                    "feature_layout": feature_layout,
                    "categorical_features": categorical_features,
                    "binary_features": binary_features,
                    "numeric_features": numeric_features,
                    "vocabulary_sizes": vocabulary_sizes,
                },
                best_model_path,
                compress=3,
            )
            print(
                f"New best model saved at epoch {epoch}: "
                f"LogLoss={best_validation_loss:.6f}"
            )
        else:
            print(
                f"Validation LogLoss did not improve; "
                f"best epoch remains {best_epoch}."
            )

    print("\n" + "=" * 80)
    print(f"FINAL EVALUATION USING BEST EPOCH {best_epoch}")
    print("=" * 80)
    checkpoint = joblib.load(best_model_path)
    best_model: SGDClassifier = checkpoint["model"]
    best_scaler: StandardScaler = checkpoint["scaler"]

    final_metrics: dict[str, dict] = {}
    evaluation_splits = (
        ("val", "test") if args.evaluate_test else ("val",)
    )
    for split_name in evaluation_splits:
        final_metrics[split_name] = evaluate_split(
            model=best_model,
            split_name=split_name,
            columns=model_columns,
            categorical_features=categorical_features,
            binary_features=binary_features,
            numeric_features=numeric_features,
            vocabulary_sizes=vocabulary_sizes,
            feature_layout=feature_layout,
            scaler=best_scaler,
            batch_size=batch_size,
            max_rows=max_eval_rows,
            prediction_path=(
                prediction_dir / f"{split_name}_predictions.parquet"
            ),
        )

    metrics_document = {
        "run_name": run_name,
        "mode": "smoke_test" if args.smoke_test else "full",
        "feature_set": args.feature_set,
        "best_epoch": best_epoch,
        "selection_metric": "validation_log_loss",
        "hyperparameters": {
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "alpha": args.alpha,
            "loss": "log_loss",
            "penalty": "l2",
            "learning_rate": "constant",
            "eta0": args.eta0,
            "average_sgd_start": 1_000_000,
            "class_weight": None,
            "random_seed": args.seed,
        },
        "validation": final_metrics["val"],
        "test": final_metrics.get("test"),
    }

    with (result_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics_document, file, indent=2)

    metric_rows = []
    for split_name in evaluation_splits:
        metric_rows.append(
            {
                "model": run_name,
                "split": split_name,
                **final_metrics[split_name],
            }
        )
    write_csv(result_dir / "metrics.csv", metric_rows)

    run_manifest = {
        "run_name": run_name,
        "feature_set": args.feature_set,
        "input_directory": str(INPUT_DIR),
        "evaluated_test": args.evaluate_test,
        "encoding_manifest": str(ENCODING_MANIFEST_PATH),
        "model_path": str(best_model_path),
        "prediction_directory": str(prediction_dir),
        "feature_layout": feature_layout,
        "categorical_features": categorical_features,
        "binary_features": binary_features,
        "numeric_features": numeric_features,
        "excluded_features": sorted(excluded_features),
        "numeric_scaler_mean": best_scaler.mean_.tolist(),
        "numeric_scaler_scale": best_scaler.scale_.tolist(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pyarrow": pa.__version__,
        },
    }
    with (result_dir / "run_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_manifest, file, indent=2)

    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION COMPLETED")
    print("=" * 80)
    print(f"Best model:   {best_model_path}")
    print(f"Metrics:      {result_dir / 'metrics.csv'}")
    print(f"Predictions:  {prediction_dir}")
    if args.smoke_test:
        print(
            "\nSmoke test passed. Run without --smoke-test for the "
            "complete experiment."
        )


def main() -> None:
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
