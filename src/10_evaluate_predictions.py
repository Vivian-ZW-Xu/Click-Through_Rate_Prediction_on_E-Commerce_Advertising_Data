#!/usr/bin/env python3
"""Shared evaluation for every CTR model.

The evaluator never trains or calibrates a model.  It validates saved
predictions, joins them back to raw impression identities, and emits one
consistent set of ranking, probability, calibration, GAUC, and cold-start
reports.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "processed" / "features"
PREDICTION_ROOT = ROOT / "artifacts" / "predictions"
ENCODER_MAPPING_DIR = ROOT / "artifacts" / "encoders" / "mappings"
RESULT_ROOT = ROOT / "results"
SPLITS = ("val", "test")
EPS = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one model run with the shared CTR protocol."
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run directory under artifacts/predictions, e.g. logistic_regression_no_weekday.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of equal-count reliability bins (default: 10).",
    )
    parser.add_argument(
        "--prediction-stage",
        choices=("raw", "calibrated"),
        default="raw",
        help=(
            "Evaluate raw model predictions (default) or the calibrated "
            "frozen-test predictions produced by src/14_calibrate_predictions.py."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("--run-name contains unsupported characters")
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else math.nan


def metric_row(split: str, segment: str, frame: pl.DataFrame) -> dict[str, object]:
    y = frame["clk"].to_numpy().astype(np.uint8, copy=False)
    p = frame["prediction"].to_numpy().astype(np.float64, copy=False)
    p_clip = np.clip(p, EPS, 1.0 - EPS)
    actual_ctr = float(y.mean())
    mean_prediction = float(p.mean())
    return {
        "split": split,
        "segment": segment,
        "rows": frame.height,
        "clicks": int(y.sum()),
        "actual_ctr": actual_ctr,
        "mean_prediction": mean_prediction,
        "auc": safe_auc(y, p),
        "logloss": float(log_loss(y, p_clip, labels=[0, 1])),
        "pr_auc": float(average_precision_score(y, p)),
        "copc": actual_ctr / mean_prediction if mean_prediction > 0 else math.nan,
        "prediction_bias_pct": (
            100.0 * (mean_prediction / actual_ctr - 1.0)
            if actual_ctr > 0
            else math.nan
        ),
    }


def load_and_validate(
    split: str,
    run_name: str,
    prediction_stage: str,
) -> pl.DataFrame:
    if prediction_stage == "calibrated":
        prediction_path = (
            PREDICTION_ROOT
            / run_name
            / "calibration"
            / f"{split}_predictions.parquet"
        )
        prediction_column = "calibrated_prediction"
    else:
        prediction_path = PREDICTION_ROOT / run_name / f"{split}_predictions.parquet"
        prediction_column = "prediction"
    feature_path = FEATURE_DIR / f"{split}.parquet"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing predictions: {prediction_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature table: {feature_path}")

    predictions = pl.read_parquet(prediction_path).select(
        pl.col("impression_id"),
        pl.col("clk").alias("prediction_clk"),
        pl.col(prediction_column).cast(pl.Float64).alias("prediction"),
    )
    if predictions["impression_id"].n_unique() != predictions.height:
        raise ValueError(f"{split}: duplicate impression_id values in predictions")
    invalid_predictions = predictions.filter(
        pl.col("prediction").is_null()
        | pl.col("prediction").is_nan()
        | pl.col("prediction").is_infinite()
        | (pl.col("prediction") < 0.0)
        | (pl.col("prediction") > 1.0)
    ).height
    if invalid_predictions:
        raise ValueError(f"{split}: found {invalid_predictions:,} invalid probabilities")

    context = pl.read_parquet(
        feature_path,
        columns=[
            "impression_id",
            "clk",
            "user_raw",
            "adgroup_id_raw",
            "pid_raw",
            "hour",
            "event_date_bj",
        ],
    )
    if context["impression_id"].n_unique() != context.height:
        raise ValueError(f"{split}: duplicate impression_id values in feature table")

    joined = predictions.join(
        context,
        on="impression_id",
        how="left",
        validate="1:1",
    )
    missing_context = joined["user_raw"].null_count()
    if joined.height != predictions.height or missing_context:
        raise ValueError(
            f"{split}: prediction/context join failed; "
            f"rows={joined.height:,}, missing_context={missing_context:,}"
        )
    label_mismatches = joined.filter(pl.col("prediction_clk") != pl.col("clk")).height
    if label_mismatches:
        raise ValueError(f"{split}: {label_mismatches:,} saved labels disagree with source data")

    return joined.drop("prediction_clk")


def build_reliability(split: str, frame: pl.DataFrame, bins: int) -> pl.DataFrame:
    n = frame.height
    ordered = frame.select("clk", "prediction").sort("prediction").with_row_index("rank")
    return (
        ordered.with_columns(
            ((pl.col("rank").cast(pl.Int64) * bins // n) + 1)
            .clip(1, bins)
            .cast(pl.Int8)
            .alias("prediction_bin")
        )
        .group_by("prediction_bin")
        .agg(
            pl.len().alias("rows"),
            pl.col("clk").sum().alias("clicks"),
            pl.col("clk").mean().alias("actual_ctr"),
            pl.col("prediction").mean().alias("mean_prediction"),
            pl.col("prediction").min().alias("min_prediction"),
            pl.col("prediction").max().alias("max_prediction"),
        )
        .with_columns(
            pl.lit(split).alias("split"),
            (pl.col("actual_ctr") / pl.col("mean_prediction")).alias("copc"),
            (pl.col("actual_ctr") - pl.col("mean_prediction")).abs().alias(
                "absolute_calibration_error"
            ),
        )
        .select(
            "split",
            "prediction_bin",
            "rows",
            "clicks",
            "actual_ctr",
            "mean_prediction",
            "copc",
            "absolute_calibration_error",
            "min_prediction",
            "max_prediction",
        )
        .sort("prediction_bin")
    )


def build_gauc(split: str, frame: pl.DataFrame) -> tuple[dict[str, object], pl.DataFrame]:
    # AUC equals the Mann-Whitney U statistic.  Average ranks make this exact
    # even when predictions tie, without looping over hundreds of thousands of users.
    per_user = (
        frame.select("user_raw", "clk", "prediction")
        .with_columns(
            pl.col("prediction").rank(method="average").over("user_raw").alias("rank")
        )
        .group_by("user_raw")
        .agg(
            pl.len().alias("rows"),
            pl.col("clk").sum().cast(pl.Int64).alias("positive_rows"),
            pl.col("rank").filter(pl.col("clk") == 1).sum().alias("positive_rank_sum"),
        )
        .with_columns((pl.col("rows") - pl.col("positive_rows")).alias("negative_rows"))
        .with_columns(
            (
                (
                    pl.col("positive_rank_sum")
                    - pl.col("positive_rows") * (pl.col("positive_rows") + 1) / 2
                )
                / (pl.col("positive_rows") * pl.col("negative_rows"))
            ).alias("user_auc")
        )
    )
    eligible = per_user.filter(
        (pl.col("positive_rows") > 0) & (pl.col("negative_rows") > 0)
    )
    eligible_rows = int(eligible["rows"].sum()) if eligible.height else 0
    total_users = per_user.height
    if eligible_rows:
        gauc = float((eligible["user_auc"] * eligible["rows"]).sum() / eligible_rows)
    else:
        gauc = math.nan
    summary = {
        "split": split,
        "gauc": gauc,
        "total_users": total_users,
        "eligible_users": eligible.height,
        "eligible_user_rate": eligible.height / total_users if total_users else math.nan,
        "total_rows": frame.height,
        "eligible_rows": eligible_rows,
        "eligible_row_coverage": eligible_rows / frame.height if frame.height else math.nan,
        "weighting": "impression_count",
    }
    return summary, eligible.with_columns(pl.lit(split).alias("split")).select(
        "split",
        "user_raw",
        "rows",
        "positive_rows",
        "negative_rows",
        "user_auc",
    )


def load_known_keys(filename: str, raw_column: str) -> pl.DataFrame:
    path = ENCODER_MAPPING_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing training vocabulary {path}. Run src/07_build_model_inputs.py first."
        )
    schema_names = pl.read_parquet_schema(path).names()
    if raw_column not in schema_names:
        raise ValueError(f"{path} does not contain {raw_column}; columns={schema_names}")
    return pl.read_parquet(path, columns=[raw_column]).unique()


def add_cold_start_flags(frame: pl.DataFrame) -> pl.DataFrame:
    known_users = load_known_keys("user_raw_mapping.parquet", "user_raw").with_columns(
        pl.lit(True).alias("known_user")
    )
    known_ads = load_known_keys("adgroup_id_raw_mapping.parquet", "adgroup_id_raw").with_columns(
        pl.lit(True).alias("known_ad")
    )
    return (
        frame.join(known_users, on="user_raw", how="left", validate="m:1")
        .join(known_ads, on="adgroup_id_raw", how="left", validate="m:1")
        .with_columns(
            pl.col("known_user").fill_null(False),
            pl.col("known_ad").fill_null(False),
        )
        .with_columns(
            pl.when(pl.col("known_user") & pl.col("known_ad"))
            .then(pl.lit("known_user__known_ad"))
            .when(pl.col("known_user") & ~pl.col("known_ad"))
            .then(pl.lit("known_user__new_ad"))
            .when(~pl.col("known_user") & pl.col("known_ad"))
            .then(pl.lit("new_user__known_ad"))
            .otherwise(pl.lit("new_user__new_ad"))
            .alias("cold_start_segment")
        )
    )


def build_cold_start(split: str, frame: pl.DataFrame) -> pl.DataFrame:
    flagged = add_cold_start_flags(frame)
    rows = []
    for key, segment_frame in flagged.partition_by(
        "cold_start_segment", as_dict=True, maintain_order=True
    ).items():
        segment = key[0] if isinstance(key, tuple) else key
        rows.append(metric_row(split, str(segment), segment_frame))
    return pl.DataFrame(rows).sort("segment")


def weighted_ece(reliability: pl.DataFrame) -> float:
    return float(
        (
            reliability["absolute_calibration_error"] * reliability["rows"]
        ).sum()
        / reliability["rows"].sum()
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.prediction_stage == "calibrated":
        output_dir = RESULT_ROOT / args.run_name / "calibration" / "evaluation"
        splits = ("test",)
        prediction_source = PREDICTION_ROOT / args.run_name / "calibration"
    else:
        output_dir = RESULT_ROOT / args.run_name / "evaluation"
        splits = SPLITS
        prediction_source = PREDICTION_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SHARED CTR EVALUATION")
    print("=" * 78)
    print(f"Run:  {args.run_name}")
    print(f"Prediction stage: {args.prediction_stage}")

    overall_rows: list[dict[str, object]] = []
    gauc_rows: list[dict[str, object]] = []
    reliability_tables: list[pl.DataFrame] = []
    cold_start_tables: list[pl.DataFrame] = []
    per_user_tables: list[pl.DataFrame] = []

    for split in splits:
        print(f"\nLoading and validating {split}...")
        frame = load_and_validate(
            split,
            args.run_name,
            args.prediction_stage,
        )
        overall = metric_row(split, "overall", frame)
        reliability = build_reliability(split, frame, args.bins)
        gauc_summary, per_user = build_gauc(split, frame)
        cold_start = build_cold_start(split, frame)
        overall["ece"] = weighted_ece(reliability)

        overall_rows.append(overall)
        gauc_rows.append(gauc_summary)
        reliability_tables.append(reliability)
        cold_start_tables.append(cold_start)
        per_user_tables.append(per_user)

        print(
            f"  rows={frame.height:,} "
            f"AUC={overall['auc']:.6f} "
            f"LogLoss={overall['logloss']:.6f} "
            f"GAUC={gauc_summary['gauc']:.6f} "
            f"COPC={overall['copc']:.6f} "
            f"ECE={overall['ece']:.6f}"
        )

    overall_df = pl.DataFrame(overall_rows)
    gauc_df = pl.DataFrame(gauc_rows)
    reliability_df = pl.concat(reliability_tables)
    cold_start_df = pl.concat(cold_start_tables)
    per_user_df = pl.concat(per_user_tables)

    overall_df.write_csv(output_dir / "overall_metrics.csv")
    gauc_df.write_csv(output_dir / "gauc_summary.csv")
    reliability_df.write_csv(output_dir / "reliability_bins.csv")
    cold_start_df.write_csv(output_dir / "cold_start_metrics.csv")
    per_user_df.write_parquet(output_dir / "eligible_user_auc.parquet", compression="zstd")

    manifest = {
        "run_name": args.run_name,
        "prediction_stage": args.prediction_stage,
        "splits": list(splits),
        "reliability_bins": args.bins,
        "gauc_weighting": "impression_count",
        "cold_start_definition": "raw user/ad ID absent from train-only encoder vocabulary",
        "prediction_source": str(prediction_source),
        "feature_source": str(FEATURE_DIR),
        "outputs": [
            "overall_metrics.csv",
            "gauc_summary.csv",
            "reliability_bins.csv",
            "cold_start_metrics.csv",
            "eligible_user_auc.parquet",
        ],
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nOverall metrics:")
    print(overall_df)
    print("\nGAUC coverage:")
    print(gauc_df)
    print("\nCold-start quadrants:")
    print(cold_start_df)
    print(f"\nSaved shared evaluation to: {output_dir}")


if __name__ == "__main__":
    main()
