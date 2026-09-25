from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


EPSILON = 1e-7
DEFAULT_BINS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit probability calibration on validation predictions and "
            "evaluate the selected calibrator on the frozen test split."
        )
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Existing prediction run under artifacts/predictions/.",
    )
    parser.add_argument(
        "--selection-folds",
        type=int,
        default=5,
        help=(
            "One deterministic validation fold is held out to select the "
            "calibration method. Default 5 gives an 80/20 split."
        ),
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of equal-width probability bins used for ECE.",
    )
    args = parser.parse_args()
    if args.selection_folds < 2:
        parser.error("--selection-folds must be at least 2")
    if args.ece_bins < 2:
        parser.error("--ece-bins must be at least 2")
    return args


def prediction_path(run_name: str, split: str) -> Path:
    return (
        Path("artifacts/predictions")
        / run_name
        / f"{split}_predictions.parquet"
    )


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    frame = pl.read_parquet(
        path,
        columns=["impression_id", "clk", "prediction"],
    )
    if frame.height == 0:
        raise ValueError(f"Prediction file is empty: {path}")
    if frame["impression_id"].n_unique() != frame.height:
        raise ValueError(f"Duplicate impression_id values in {path}")

    impression_id = frame["impression_id"].to_numpy().astype(np.uint64)
    labels = frame["clk"].to_numpy().astype(np.int8)
    predictions = frame["prediction"].to_numpy().astype(np.float64)

    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Non-binary labels found in {path}")
    if not np.isfinite(predictions).all():
        raise ValueError(f"Non-finite predictions found in {path}")
    if ((predictions < 0.0) | (predictions > 1.0)).any():
        raise ValueError(f"Predictions outside [0, 1] found in {path}")

    return impression_id, labels, predictions


def clipped(probability: np.ndarray) -> np.ndarray:
    return np.clip(probability, EPSILON, 1.0 - EPSILON)


def logit_feature(probability: np.ndarray) -> np.ndarray:
    probability = clipped(probability)
    return np.log(probability / (1.0 - probability)).reshape(-1, 1)


def deterministic_selection_mask(
    impression_id: np.ndarray,
    folds: int,
) -> np.ndarray:
    hashed = impression_id * np.uint64(11400714819323198485)
    return (hashed % np.uint64(folds)) == 0


def fit_platt(labels: np.ndarray, predictions: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(
        C=1_000_000.0,
        solver="lbfgs",
        max_iter=500,
        random_state=42,
    )
    calibrator.fit(logit_feature(predictions), labels)
    return calibrator


def apply_platt(
    calibrator: LogisticRegression,
    predictions: np.ndarray,
) -> np.ndarray:
    return calibrator.predict_proba(logit_feature(predictions))[:, 1]


def fit_isotonic(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> IsotonicRegression:
    calibrator = IsotonicRegression(
        y_min=EPSILON,
        y_max=1.0 - EPSILON,
        out_of_bounds="clip",
    )
    calibrator.fit(predictions, labels)
    return calibrator


def expected_calibration_error(
    labels: np.ndarray,
    predictions: np.ndarray,
    bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_index = np.clip(
        np.searchsorted(edges, predictions, side="right") - 1,
        0,
        bins - 1,
    )
    count = np.bincount(bin_index, minlength=bins)
    label_sum = np.bincount(bin_index, weights=labels, minlength=bins)
    prediction_sum = np.bincount(
        bin_index,
        weights=predictions,
        minlength=bins,
    )
    nonempty = count > 0
    actual = label_sum[nonempty] / count[nonempty]
    predicted = prediction_sum[nonempty] / count[nonempty]
    weights = count[nonempty] / labels.size
    return float(np.sum(weights * np.abs(actual - predicted)))


def metric_row(
    stage: str,
    method: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    bins: int,
) -> dict[str, object]:
    predictions = clipped(predictions)
    actual_ctr = float(labels.mean())
    mean_prediction = float(predictions.mean())
    copc = actual_ctr / mean_prediction
    return {
        "stage": stage,
        "method": method,
        "rows": int(labels.size),
        "clicks": int(labels.sum()),
        "actual_ctr": actual_ctr,
        "mean_prediction": mean_prediction,
        "auc": float(roc_auc_score(labels, predictions)),
        "logloss": float(log_loss(labels, predictions)),
        "pr_auc": float(average_precision_score(labels, predictions)),
        "brier_score": float(brier_score_loss(labels, predictions)),
        "copc": copc,
        "prediction_bias_pct": 100.0 * (mean_prediction / actual_ctr - 1.0),
        "ece": expected_calibration_error(labels, predictions, bins),
    }


def reliability_table(
    labels: np.ndarray,
    raw_predictions: np.ndarray,
    calibrated_predictions: np.ndarray,
    bins: int,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for method, predictions in (
        ("raw", raw_predictions),
        ("calibrated", calibrated_predictions),
    ):
        frame = pl.DataFrame(
            {
                "clk": labels,
                "prediction": clipped(predictions),
            }
        ).with_columns(
            (
                (pl.col("prediction") * bins)
                .floor()
                .clip(0, bins - 1)
                .cast(pl.Int16)
                + 1
            ).alias("probability_bin")
        )
        summary = (
            frame.group_by("probability_bin")
            .agg(
                pl.len().alias("rows"),
                pl.col("clk").sum().alias("clicks"),
                pl.col("clk").mean().alias("actual_ctr"),
                pl.col("prediction").mean().alias("mean_prediction"),
                pl.col("prediction").min().alias("min_prediction"),
                pl.col("prediction").max().alias("max_prediction"),
            )
            .with_columns(pl.lit(method).alias("method"))
            .select(
                "method",
                "probability_bin",
                "rows",
                "clicks",
                "actual_ctr",
                "mean_prediction",
                "min_prediction",
                "max_prediction",
            )
            .sort("probability_bin")
        )
        frames.append(summary)
    return pl.concat(frames)


def main() -> None:
    args = parse_args()
    run_name = args.run_name
    result_dir = Path("results") / run_name / "calibration"
    calibrator_dir = Path("artifacts/calibrators") / run_name
    prediction_dir = Path("artifacts/predictions") / run_name / "calibration"
    result_dir.mkdir(parents=True, exist_ok=True)
    calibrator_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PROBABILITY CALIBRATION")
    print("=" * 78)
    print(f"Run:                    {run_name}")
    print(f"Selection protocol:     validation {args.selection_folds - 1}/{args.selection_folds} fit, 1/{args.selection_folds} select")
    print("Frozen test selection:  disabled")

    print("\nLoading validation predictions...")
    val_id, val_y, val_p = load_predictions(prediction_path(run_name, "val"))
    selection_mask = deterministic_selection_mask(
        val_id,
        args.selection_folds,
    )
    fit_mask = ~selection_mask
    print(f"  calibration-fit rows: {int(fit_mask.sum()):,}")
    print(f"  selection rows:       {int(selection_mask.sum()):,}")

    print("\nFitting candidate calibrators on calibration-fit rows...")
    platt = fit_platt(val_y[fit_mask], val_p[fit_mask])
    isotonic = fit_isotonic(val_y[fit_mask], val_p[fit_mask])

    selection_predictions = {
        "raw": val_p[selection_mask],
        "platt": apply_platt(platt, val_p[selection_mask]),
        "isotonic": isotonic.predict(val_p[selection_mask]),
    }
    selection_rows = [
        metric_row(
            stage="validation_selection",
            method=method,
            labels=val_y[selection_mask],
            predictions=predictions,
            bins=args.ece_bins,
        )
        for method, predictions in selection_predictions.items()
    ]
    selection_metrics = pl.DataFrame(selection_rows).sort("logloss")
    print("\nValidation selection metrics:")
    print(selection_metrics)

    calibrated_candidates = selection_metrics.filter(
        pl.col("method") != "raw"
    )
    selected_method = calibrated_candidates.row(0, named=True)["method"]
    print(f"\nSelected calibration method by validation LogLoss: {selected_method}")

    print("Refitting selected calibrator on the complete validation split...")
    if selected_method == "platt":
        final_calibrator = fit_platt(val_y, val_p)
        apply_final = lambda values: apply_platt(final_calibrator, values)
    else:
        final_calibrator = fit_isotonic(val_y, val_p)
        apply_final = final_calibrator.predict

    calibrator_path = calibrator_dir / f"{selected_method}.joblib"
    joblib.dump(final_calibrator, calibrator_path)

    print("\nLoading frozen test predictions...")
    test_id, test_y, test_p = load_predictions(prediction_path(run_name, "test"))
    test_calibrated = clipped(apply_final(test_p))

    test_rows = [
        metric_row(
            stage="frozen_test",
            method="raw",
            labels=test_y,
            predictions=test_p,
            bins=args.ece_bins,
        ),
        metric_row(
            stage="frozen_test",
            method=selected_method,
            labels=test_y,
            predictions=test_calibrated,
            bins=args.ece_bins,
        ),
    ]
    test_metrics = pl.DataFrame(test_rows)
    print("\nFrozen test metrics:")
    print(test_metrics)

    selection_metrics.write_csv(result_dir / "selection_metrics.csv")
    test_metrics.write_csv(result_dir / "test_metrics.csv")
    reliability_table(
        test_y,
        test_p,
        test_calibrated,
        args.ece_bins,
    ).write_csv(result_dir / "test_reliability_bins.csv")

    pl.DataFrame(
        {
            "impression_id": test_id,
            "clk": test_y,
            "raw_prediction": test_p.astype(np.float32),
            "calibrated_prediction": test_calibrated.astype(np.float32),
        }
    ).write_parquet(
        prediction_dir / "test_predictions.parquet",
        compression="zstd",
    )

    manifest = {
        "run_name": run_name,
        "selected_method": selected_method,
        "selection_rule": "lowest LogLoss on deterministic validation holdout",
        "selection_folds": args.selection_folds,
        "calibration_fit_rows": int(fit_mask.sum()),
        "selection_rows": int(selection_mask.sum()),
        "full_validation_refit_rows": int(val_y.size),
        "frozen_test_rows": int(test_y.size),
        "ece_bins": args.ece_bins,
        "probability_clip": EPSILON,
        "calibrator_path": str(calibrator_path),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with (result_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("\n" + "=" * 78)
    print("CALIBRATION COMPLETED")
    print("=" * 78)
    print(f"Selected method: {selected_method}")
    print(f"Calibrator:      {calibrator_path}")
    print(f"Results:         {result_dir}")
    print(f"Predictions:     {prediction_dir}")


if __name__ == "__main__":
    main()
