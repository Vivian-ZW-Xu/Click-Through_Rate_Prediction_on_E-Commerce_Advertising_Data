from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


INPUT_DIR = Path("data/processed/model_input_f3")
BASE_MANIFEST_PATH = Path("artifacts/encoders/encoding_manifest.json")
F3_MANIFEST_PATH = Path("results/model_input_f3/feature_manifest.json")

DEFAULT_RUN_NAME = "lightgbm_f1_f2_f3"
DEFAULT_NUM_THREADS = 14
DEFAULT_NUM_BOOST_ROUND = 1_000
DEFAULT_EARLY_STOPPING_ROUNDS = 50

SMOKE_TRAIN_ROWS = 1_000_000
SMOKE_VAL_ROWS = 300_000
SMOKE_NUM_BOOST_ROUND = 100
SMOKE_EARLY_STOPPING_ROUNDS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a memory-aware LightGBM CTR model on the complete "
            "F1+F2+F3 information budget."
        )
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Generate frozen-test predictions after model selection.",
    )
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=DEFAULT_NUM_BOOST_ROUND,
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_EARLY_STOPPING_ROUNDS,
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-data-in-leaf", type=int, default=2_000)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--cat-l2", type=float, default=10.0)
    parser.add_argument("--cat-smooth", type=float, default=20.0)
    parser.add_argument("--max-cat-threshold", type=int, default=32)
    parser.add_argument("--min-data-per-group", type=int, default=100)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        parser.error(
            "--run-name may contain only letters, numbers, dot, "
            "underscore, and hyphen"
        )
    if args.num_threads < 1:
        parser.error("--num-threads must be positive")
    if args.num_boost_round < 1:
        parser.error("--num-boost-round must be positive")
    if args.early_stopping_rounds < 1:
        parser.error("--early-stopping-rounds must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.num_leaves < 2:
        parser.error("--num-leaves must be at least 2")
    if args.min_data_in_leaf < 1:
        parser.error("--min-data-in-leaf must be positive")
    for name in ("feature_fraction", "bagging_fraction"):
        value = getattr(args, name)
        if not 0 < value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    if args.lambda_l1 < 0 or args.lambda_l2 < 0:
        parser.error("regularization strengths cannot be negative")
    if args.cat_l2 < 0 or args.cat_smooth < 0:
        parser.error("categorical regularization cannot be negative")
    return args


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required manifest not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def feature_schema() -> tuple[list[str], list[str], list[str]]:
    base = load_json(BASE_MANIFEST_PATH)
    f3 = load_json(F3_MANIFEST_PATH)

    categorical = [
        feature
        for feature in base["categorical_features"]
        if feature != "day_of_week_idx"
    ]
    binary = [
        *base["binary_features"],
        *f3["binary_features"],
    ]
    numeric = [
        *base["numeric_features"],
        *f3["numeric_features"],
    ]

    all_features = [*categorical, *binary, *numeric]
    duplicate = sorted(
        feature for feature in set(all_features) if all_features.count(feature) > 1
    )
    if duplicate:
        raise ValueError(f"Duplicate model features: {duplicate}")
    return categorical, binary, numeric


def validate_input_schema(path: Path, required_columns: list[str]) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Model input not found: {path}. Run src/12_build_f3_model_inputs.py first."
        )
    schema_names = set(pl.scan_parquet(path).collect_schema().names())
    missing = sorted(set(required_columns) - schema_names)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def load_matrix(
    split: str,
    features: list[str],
    row_limit: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = INPUT_DIR / f"{split}.parquet"
    required = ["impression_id", "clk", *features]
    validate_input_schema(path, required)

    query = pl.scan_parquet(path).select(required)
    if row_limit is not None:
        query = query.head(row_limit)

    started = time.perf_counter()
    frame = query.collect()
    impression_id = frame["impression_id"].to_numpy().astype(
        np.uint32,
        copy=False,
    )
    labels = frame["clk"].to_numpy().astype(np.uint8, copy=False)

    feature_frame = frame.select(
        pl.col(feature).cast(pl.Float32).alias(feature)
        for feature in features
    )
    matrix = np.ascontiguousarray(feature_frame.to_numpy(), dtype=np.float32)

    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Non-binary labels found in {path}")
    if impression_id.size != np.unique(impression_id).size:
        raise ValueError(f"Duplicate impression IDs found in {path}")

    elapsed = time.perf_counter() - started
    print(
        f"  loaded {split}: rows={matrix.shape[0]:,}, "
        f"features={matrix.shape[1]}, matrix={matrix.nbytes / 2**30:.2f} GiB, "
        f"time={elapsed / 60:.1f}m"
    )
    del feature_frame, frame
    gc.collect()
    return impression_id, labels, matrix


def build_dataset(
    split: str,
    features: list[str],
    categorical_indices: list[int],
    row_limit: int | None,
    reference: lgb.Dataset | None = None,
) -> tuple[lgb.Dataset, int]:
    _, labels, matrix = load_matrix(split, features, row_limit)
    row_count = labels.size
    dataset = lgb.Dataset(
        matrix,
        label=labels,
        feature_name=features,
        categorical_feature=categorical_indices,
        reference=reference,
        free_raw_data=True,
    )
    print(f"  constructing LightGBM {split} Dataset...")
    dataset.construct()
    del matrix, labels
    gc.collect()
    return dataset, row_count


def metric_row(
    model_name: str,
    split: str,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    predictions = np.clip(predictions.astype(np.float64), 1e-7, 1 - 1e-7)
    return {
        "model": model_name,
        "split": split,
        "rows": int(labels.size),
        "clicks": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "mean_prediction": float(predictions.mean()),
        "auc": float(roc_auc_score(labels, predictions)),
        "log_loss": float(log_loss(labels, predictions)),
        "pr_auc": float(average_precision_score(labels, predictions)),
        "brier_score": float(brier_score_loss(labels, predictions)),
    }


def predict_and_save(
    booster: lgb.Booster,
    run_name: str,
    split: str,
    features: list[str],
    row_limit: int | None,
    prediction_dir: Path,
) -> dict[str, object]:
    impression_id, labels, matrix = load_matrix(split, features, row_limit)
    started = time.perf_counter()
    predictions = booster.predict(
        matrix,
        num_iteration=booster.best_iteration,
    )
    elapsed = time.perf_counter() - started
    row = metric_row(run_name, split, labels, predictions)
    print(
        f"  {split:<5} rows={row['rows']:,} AUC={row['auc']:.6f} "
        f"LogLoss={row['log_loss']:.6f} PR-AUC={row['pr_auc']:.6f} "
        f"mean(p)={row['mean_prediction']:.6f} time={elapsed / 60:.1f}m"
    )

    pl.DataFrame(
        {
            "impression_id": impression_id,
            "clk": labels,
            "prediction": predictions.astype(np.float32),
        }
    ).write_parquet(
        prediction_dir / f"{split}_predictions.parquet",
        compression="zstd",
    )
    del impression_id, labels, matrix, predictions
    gc.collect()
    return row


def main() -> None:
    args = parse_args()
    run_name = args.run_name + ("_smoke" if args.smoke_test else "")

    categorical, binary, numeric = feature_schema()
    features = [*categorical, *binary, *numeric]
    categorical_indices = list(range(len(categorical)))

    train_limit = SMOKE_TRAIN_ROWS if args.smoke_test else None
    val_limit = SMOKE_VAL_ROWS if args.smoke_test else None
    test_limit = SMOKE_VAL_ROWS if args.smoke_test else None
    num_boost_round = (
        SMOKE_NUM_BOOST_ROUND if args.smoke_test else args.num_boost_round
    )
    early_stopping_rounds = (
        SMOKE_EARLY_STOPPING_ROUNDS
        if args.smoke_test
        else args.early_stopping_rounds
    )

    result_dir = Path("results") / run_name
    model_dir = Path("artifacts/models") / run_name
    prediction_dir = Path("artifacts/predictions") / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "cat_l2": args.cat_l2,
        "cat_smooth": args.cat_smooth,
        "max_cat_threshold": args.max_cat_threshold,
        "min_data_per_group": args.min_data_per_group,
        "max_bin": args.max_bin,
        "num_threads": args.num_threads,
        "seed": args.seed,
        "feature_fraction_seed": args.seed,
        "bagging_seed": args.seed,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }

    print("=" * 80)
    print("LIGHTGBM CTR TRAINING")
    print("=" * 80)
    print(f"Run:                    {run_name}")
    print("Feature set:            F1+F2+F3")
    print(f"Categorical features:   {len(categorical)}")
    print(f"Binary features:        {len(binary)}")
    print(f"Numeric features:       {len(numeric)}")
    print(f"Total features:         {len(features)}")
    print(f"CPU threads:            {args.num_threads}")
    print(f"Maximum boosting rounds:{num_boost_round:>9}")
    print(f"Early stopping rounds:  {early_stopping_rounds}")
    print(f"Evaluate frozen test:   {args.evaluate_test}")
    if args.smoke_test:
        print(f"Smoke train rows:       {train_limit:,}")
        print(f"Smoke validation rows:  {val_limit:,}")

    started = time.perf_counter()
    print("\nBuilding training Dataset...")
    train_set, train_rows = build_dataset(
        "train",
        features,
        categorical_indices,
        train_limit,
    )
    print("\nBuilding validation Dataset...")
    val_set, val_rows = build_dataset(
        "val",
        features,
        categorical_indices,
        val_limit,
        reference=train_set,
    )

    evaluation_history: dict[str, dict[str, list[float]]] = {}
    print("\nTraining LightGBM...")
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(
                early_stopping_rounds,
                first_metric_only=True,
                verbose=True,
            ),
            lgb.log_evaluation(period=25),
            lgb.record_evaluation(evaluation_history),
        ],
    )
    training_minutes = (time.perf_counter() - started) / 60
    print(f"Best iteration: {booster.best_iteration}")
    print(f"Training and Dataset construction time: {training_minutes:.1f}m")

    model_path = model_dir / "best_model.txt"
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    history = pl.DataFrame(
        {
            "iteration": np.arange(
                1,
                len(evaluation_history["val"]["binary_logloss"]) + 1,
            ),
            "val_log_loss": evaluation_history["val"]["binary_logloss"],
            "val_auc": evaluation_history["val"]["auc"],
        }
    )
    history.write_csv(result_dir / "training_history.csv")

    importance = pl.DataFrame(
        {
            "feature": features,
            "gain": booster.feature_importance(
                importance_type="gain",
                iteration=booster.best_iteration,
            ),
            "split": booster.feature_importance(
                importance_type="split",
                iteration=booster.best_iteration,
            ),
        }
    ).with_columns(
        (
            pl.col("gain") / pl.col("gain").sum()
        ).alias("gain_fraction")
    ).sort("gain", descending=True)
    importance.write_csv(result_dir / "feature_importance.csv")

    del train_set, val_set
    gc.collect()

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    metric_rows = [
        predict_and_save(
            booster,
            run_name,
            "val",
            features,
            val_limit,
            prediction_dir,
        )
    ]
    if args.evaluate_test:
        metric_rows.append(
            predict_and_save(
                booster,
                run_name,
                "test",
                features,
                test_limit,
                prediction_dir,
            )
        )

    metrics = pl.DataFrame(metric_rows)
    metrics.write_csv(result_dir / "metrics.csv")
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metric_rows, file, indent=2)

    manifest = {
        "run_name": run_name,
        "model_family": "LightGBM GBDT",
        "feature_set": "F1+F2+F3",
        "input_directory": str(INPUT_DIR),
        "categorical_features": categorical,
        "binary_features": binary,
        "numeric_features": numeric,
        "excluded_features": ["day_of_week_idx"],
        "categorical_handling": "LightGBM native categorical splits",
        "matrix_dtype": "float32",
        "train_rows": train_rows,
        "validation_rows": val_rows,
        "smoke_test": args.smoke_test,
        "evaluate_test": args.evaluate_test,
        "best_iteration": booster.best_iteration,
        "training_minutes_including_dataset_construction": training_minutes,
        "params": params,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "lightgbm": lgb.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with (result_dir / "run_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("\nTop 15 features by gain:")
    print(importance.head(15))
    print("\n" + "=" * 80)
    print("LIGHTGBM COMPLETED")
    print("=" * 80)
    print(f"Best model:   {model_path}")
    print(f"Metrics:      {result_dir / 'metrics.csv'}")
    print(f"Predictions:  {prediction_dir}")
    if args.smoke_test:
        print("\nSmoke test passed. Do not use smoke metrics as final evidence.")


if __name__ == "__main__":
    main()
