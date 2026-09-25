#!/usr/bin/env python3
"""Attach past-only CTR features to the common encoded model inputs.

The existing data/processed/model_input directory remains the F1+F2 control.
This script writes a separate model_input_f3 directory for F1+F2+F3, making
the information ablation explicit and reversible.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
BASE_INPUT_DIR = ROOT / "data" / "processed" / "model_input"
PAST_CTR_DIR = ROOT / "data" / "processed" / "past_ctr"
OUTPUT_DIR = ROOT / "data" / "processed" / "model_input_f3"
RESULT_DIR = ROOT / "results" / "model_input_f3"
SPLITS = ("train", "val", "test")
ENTITIES = ("user", "ad", "category", "campaign", "customer", "brand")

F3_NUMERIC_FEATURES = ["global_past_ctr"] + [
    feature
    for entity in ENTITIES
    for feature in (f"{entity}_past_ctr", f"{entity}_past_impressions_log")
]
F3_BINARY_FEATURES = [f"{entity}_has_past_feedback" for entity in ENTITIES]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path


def build_split(split: str) -> None:
    base_path = require_file(BASE_INPUT_DIR / f"{split}.parquet")
    ctr_path = require_file(PAST_CTR_DIR / f"{split}.parquet")
    output_path = OUTPUT_DIR / f"{split}.parquet"

    ctr_schema = pl.scan_parquet(ctr_path).collect_schema().names()
    required_ctr = ["impression_id", "global_past_ctr"]
    for entity in ENTITIES:
        required_ctr.extend(
            [
                f"{entity}_past_impressions",
                f"{entity}_past_ctr",
                f"{entity}_has_past_feedback",
            ]
        )
    missing = sorted(set(required_ctr) - set(ctr_schema))
    if missing:
        raise ValueError(f"{ctr_path} is missing F3 columns: {missing}")

    ctr_features = pl.scan_parquet(ctr_path).select(required_ctr).with_columns(
        *[
            pl.col(f"{entity}_past_impressions")
            .cast(pl.Float64)
            .log1p()
            .cast(pl.Float32)
            .alias(f"{entity}_past_impressions_log")
            for entity in ENTITIES
        ]
    ).select(
        "impression_id",
        *F3_NUMERIC_FEATURES,
        *F3_BINARY_FEATURES,
    )

    print(f"Joining F3 into {split}...")
    (
        pl.scan_parquet(base_path)
        .join(ctr_features, on="impression_id", how="left")
        .sink_parquet(output_path, compression="zstd")
    )
    print(f"  Saved {output_path}: {output_path.stat().st_size / 1024**2:,.1f} MB")


def validate_split(split: str) -> dict[str, object]:
    base_path = BASE_INPUT_DIR / f"{split}.parquet"
    output_path = OUTPUT_DIR / f"{split}.parquet"
    base_rows = pl.scan_parquet(base_path).select(pl.len()).collect().item()
    scan = pl.scan_parquet(output_path)

    summary = scan.select(
        pl.len().alias("rows"),
        pl.col("impression_id").n_unique().alias("unique_impressions"),
        *[
            pl.col(feature).null_count().alias(f"null_{feature}")
            for feature in F3_NUMERIC_FEATURES + F3_BINARY_FEATURES
        ],
        *[
            pl.col(feature)
            .cast(pl.Float64)
            .is_nan()
            .sum()
            .alias(f"nan_{feature}")
            for feature in F3_NUMERIC_FEATURES
        ],
    ).collect()

    rows = int(summary["rows"][0])
    unique_rows = int(summary["unique_impressions"][0])
    if rows != base_rows:
        raise AssertionError(
            f"{split}: base has {base_rows:,} rows but F3 input has {rows:,}"
        )
    if unique_rows != rows:
        raise AssertionError(f"{split}: impression_id is not unique after F3 join")

    problem_columns = [
        column
        for column in summary.columns
        if (column.startswith("null_") or column.startswith("nan_"))
        and int(summary[column][0]) != 0
    ]
    if problem_columns:
        raise AssertionError(f"{split}: invalid F3 values in {problem_columns}")

    return {
        "split": split,
        "rows": rows,
        "unique_impressions": unique_rows,
        "f3_numeric_features": len(F3_NUMERIC_FEATURES),
        "f3_binary_features": len(F3_BINARY_FEATURES),
        "output_mb": output_path.stat().st_size / 1024**2,
    }


def history_count_profile() -> pl.DataFrame:
    """Report how much evidence supports each entity's smoothed CTR."""
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        path = PAST_CTR_DIR / f"{split}.parquet"
        expressions: list[pl.Expr] = [pl.len().alias("rows")]
        for entity in ENTITIES:
            column = f"{entity}_past_impressions"
            expressions.extend(
                [
                    (pl.col(column) > 0).mean().alias(f"{entity}__coverage"),
                    pl.col(column).median().alias(f"{entity}__median_all"),
                    pl.col(column).quantile(0.90).alias(f"{entity}__p90_all"),
                    pl.col(column).quantile(0.99).alias(f"{entity}__p99_all"),
                    pl.col(column).max().alias(f"{entity}__max"),
                    pl.col(column)
                    .filter(pl.col(column) > 0)
                    .median()
                    .alias(f"{entity}__median_if_present"),
                    pl.col(column)
                    .filter(pl.col(column) > 0)
                    .quantile(0.90)
                    .alias(f"{entity}__p90_if_present"),
                    (pl.col(column) < 100)
                    .mean()
                    .alias(f"{entity}__below_smoothing_strength_rate"),
                ]
            )
        stats = pl.scan_parquet(path).select(expressions).collect().row(0, named=True)
        for entity in ENTITIES:
            rows.append(
                {
                    "split": split,
                    "entity": entity,
                    "rows": stats["rows"],
                    "history_coverage": stats[f"{entity}__coverage"],
                    "median_past_impressions_all": stats[f"{entity}__median_all"],
                    "p90_past_impressions_all": stats[f"{entity}__p90_all"],
                    "p99_past_impressions_all": stats[f"{entity}__p99_all"],
                    "max_past_impressions": stats[f"{entity}__max"],
                    "median_past_impressions_if_present": stats[
                        f"{entity}__median_if_present"
                    ],
                    "p90_past_impressions_if_present": stats[
                        f"{entity}__p90_if_present"
                    ],
                    "below_smoothing_strength_rate": stats[
                        f"{entity}__below_smoothing_strength_rate"
                    ],
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("BUILDING F1 + F2 + F3 MODEL INPUTS")
    print("=" * 78)
    print(f"F3 numeric features: {len(F3_NUMERIC_FEATURES)}")
    print(f"F3 binary features:  {len(F3_BINARY_FEATURES)}")

    for split in SPLITS:
        build_split(split)

    summaries = pl.DataFrame([validate_split(split) for split in SPLITS])
    profile = history_count_profile()
    summaries.write_csv(RESULT_DIR / "model_input_f3_summary.csv")
    profile.write_csv(RESULT_DIR / "history_count_profile.csv")

    feature_manifest = {
        "base_information": "F1+F2 from data/processed/model_input",
        "added_information": "F3 past-only smoothed CTR statistics",
        "numeric_features": F3_NUMERIC_FEATURES,
        "binary_features": F3_BINARY_FEATURES,
        "excluded_raw_components": [
            f"{entity}_past_clicks" for entity in ENTITIES
        ] + [f"{entity}_past_impressions" for entity in ENTITIES],
        "count_transformation": "log1p(past impressions)",
        "reason_for_excluding_raw_clicks": (
            "past CTR plus evidence count is sufficient; raw clicks would be redundant"
        ),
    }
    (RESULT_DIR / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nValidation summary:")
    print(summaries)
    print("\nHistory evidence profile:")
    print(profile)
    print("\nAll F3 model-input validation checks passed.")
    print(f"Model inputs: {OUTPUT_DIR}")
    print(f"Reports:      {RESULT_DIR}")


if __name__ == "__main__":
    main()
