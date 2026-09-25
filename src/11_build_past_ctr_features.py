#!/usr/bin/env python3
"""Build leakage-free, past-only CTR features at daily granularity.

For an impression on Beijing date d, every statistic in this file uses only
labels from dates strictly earlier than d.  Current-day and future labels are
never included.  Outputs contain only impression_id plus F3 features, so the
existing F1+F2 tables remain unchanged for controlled ablations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "processed" / "features"
OUTPUT_DIR = ROOT / "data" / "processed" / "past_ctr"
LOOKUP_DIR = OUTPUT_DIR / "lookups"
RESULT_DIR = ROOT / "results" / "past_ctr"
SPLITS = ("train", "val", "test")
DATE_COLUMN = "event_date_bj"
LABEL_COLUMN = "clk"
DEFAULT_PRIOR = 0.05
DEFAULT_SMOOTHING = 100.0

# Prefixes are intentionally shorter than raw column names because they become
# model feature names later.
ENTITY_COLUMNS = {
    "user": "user_raw",
    "ad": "adgroup_id_raw",
    "category": "cate_id_raw",
    "campaign": "campaign_id_raw",
    "customer": "customer_raw",
    "brand": "brand_raw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build daily point-in-time CTR features from prior dates only."
    )
    parser.add_argument(
        "--smoothing-strength",
        type=float,
        default=DEFAULT_SMOOTHING,
        help="Beta-prior equivalent impression count (default: 100).",
    )
    parser.add_argument(
        "--initial-prior",
        type=float,
        default=DEFAULT_PRIOR,
        help="Prior CTR used only when no earlier labeled date exists (default: 0.05).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.smoothing_strength <= 0:
        raise ValueError("--smoothing-strength must be positive")
    if not 0 < args.initial_prior < 1:
        raise ValueError("--initial-prior must be strictly between 0 and 1")


def input_path(split: str) -> Path:
    path = INPUT_DIR / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return path


def all_impressions() -> pl.LazyFrame:
    required = [
        "impression_id",
        LABEL_COLUMN,
        DATE_COLUMN,
        *ENTITY_COLUMNS.values(),
    ]
    frames = []
    for split in SPLITS:
        path = input_path(split)
        schema_names = pl.scan_parquet(path).collect_schema().names()
        missing = sorted(set(required) - set(schema_names))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frames.append(pl.scan_parquet(path).select(required))
    return pl.concat(frames, how="vertical")


def build_global_lookup(source: pl.LazyFrame, initial_prior: float) -> pl.DataFrame:
    daily = (
        source.group_by(DATE_COLUMN)
        .agg(
            pl.len().cast(pl.UInt32).alias("daily_impressions"),
            pl.col(LABEL_COLUMN).sum().cast(pl.UInt32).alias("daily_clicks"),
        )
        .collect(engine="streaming")
        .sort(DATE_COLUMN)
    )
    return (
        daily.with_columns(
            (
                pl.col("daily_impressions").cum_sum()
                - pl.col("daily_impressions")
            )
            .cast(pl.UInt32)
            .alias("global_past_impressions"),
            (pl.col("daily_clicks").cum_sum() - pl.col("daily_clicks"))
            .cast(pl.UInt32)
            .alias("global_past_clicks"),
        )
        .with_columns(
            pl.when(pl.col("global_past_impressions") > 0)
            .then(
                pl.col("global_past_clicks").cast(pl.Float64)
                / pl.col("global_past_impressions")
            )
            .otherwise(pl.lit(initial_prior))
            .cast(pl.Float32)
            .alias("global_past_ctr")
        )
        .select(
            DATE_COLUMN,
            "global_past_impressions",
            "global_past_clicks",
            "global_past_ctr",
        )
    )


def build_entity_lookup(
    source: pl.LazyFrame,
    global_lookup: pl.DataFrame,
    prefix: str,
    entity_column: str,
    smoothing_strength: float,
) -> pl.DataFrame:
    impressions_name = f"{prefix}_past_impressions"
    clicks_name = f"{prefix}_past_clicks"
    ctr_name = f"{prefix}_past_ctr"
    history_name = f"{prefix}_has_past_feedback"

    daily = (
        source.filter(pl.col(entity_column).is_not_null())
        .group_by([entity_column, DATE_COLUMN])
        .agg(
            pl.len().cast(pl.UInt32).alias("daily_impressions"),
            pl.col(LABEL_COLUMN).sum().cast(pl.UInt32).alias("daily_clicks"),
        )
        .collect(engine="streaming")
        .sort([entity_column, DATE_COLUMN])
    )
    lookup = (
        daily.with_columns(
            (
                pl.col("daily_impressions").cum_sum().over(entity_column)
                - pl.col("daily_impressions")
            )
            .cast(pl.UInt32)
            .alias(impressions_name),
            (
                pl.col("daily_clicks").cum_sum().over(entity_column)
                - pl.col("daily_clicks")
            )
            .cast(pl.UInt32)
            .alias(clicks_name),
        )
        .join(global_lookup.select(DATE_COLUMN, "global_past_ctr"), on=DATE_COLUMN, how="left")
        .with_columns(
            (pl.col(impressions_name) > 0).alias(history_name),
            (
                (
                    pl.col(clicks_name).cast(pl.Float64)
                    + smoothing_strength * pl.col("global_past_ctr").cast(pl.Float64)
                )
                / (pl.col(impressions_name).cast(pl.Float64) + smoothing_strength)
            )
            .cast(pl.Float32)
            .alias(ctr_name),
        )
        .select(
            entity_column,
            DATE_COLUMN,
            impressions_name,
            clicks_name,
            ctr_name,
            history_name,
        )
    )
    return lookup


def write_split(
    split: str,
    global_lookup: pl.DataFrame,
) -> None:
    source = pl.scan_parquet(input_path(split)).select(
        "impression_id", DATE_COLUMN, *ENTITY_COLUMNS.values()
    )
    enriched = source.join(global_lookup.lazy(), on=DATE_COLUMN, how="left")

    feature_columns = [
        "global_past_impressions",
        "global_past_clicks",
        "global_past_ctr",
    ]
    for prefix, entity_column in ENTITY_COLUMNS.items():
        impressions_name = f"{prefix}_past_impressions"
        clicks_name = f"{prefix}_past_clicks"
        ctr_name = f"{prefix}_past_ctr"
        history_name = f"{prefix}_has_past_feedback"
        enriched = enriched.join(
            pl.scan_parquet(LOOKUP_DIR / f"{prefix}_daily_lookup.parquet"),
            on=[entity_column, DATE_COLUMN],
            how="left",
        ).with_columns(
            pl.col(impressions_name).fill_null(0).cast(pl.UInt32),
            pl.col(clicks_name).fill_null(0).cast(pl.UInt32),
            pl.col(ctr_name).fill_null(pl.col("global_past_ctr")).cast(pl.Float32),
            pl.col(history_name).fill_null(False),
        )
        feature_columns.extend(
            [impressions_name, clicks_name, ctr_name, history_name]
        )

    output_path = OUTPUT_DIR / f"{split}.parquet"
    print(f"Writing {split} -> {output_path}")
    enriched.select("impression_id", DATE_COLUMN, *feature_columns).sink_parquet(
        output_path,
        compression="zstd",
    )
    print(f"  Saved {output_path.name}: {output_path.stat().st_size / 1024**2:,.1f} MB")


def validate_outputs() -> pl.DataFrame:
    summaries = []
    expected_total = 0
    observed_total = 0
    ctr_columns = ["global_past_ctr"] + [
        f"{prefix}_past_ctr" for prefix in ENTITY_COLUMNS
    ]
    history_columns = [
        f"{prefix}_has_past_feedback" for prefix in ENTITY_COLUMNS
    ]
    count_columns = [
        f"{prefix}_past_impressions" for prefix in ENTITY_COLUMNS
    ]

    for split in SPLITS:
        source_path = input_path(split)
        output_path = OUTPUT_DIR / f"{split}.parquet"
        expected_rows = pl.scan_parquet(source_path).select(pl.len()).collect().item()
        expected_total += expected_rows

        scan = pl.scan_parquet(output_path)
        summary = (
            scan.select(
                pl.len().alias("rows"),
                pl.col("impression_id").n_unique().alias("unique_impressions"),
                pl.col(DATE_COLUMN).min().alias("min_date"),
                pl.col(DATE_COLUMN).max().alias("max_date"),
                *[
                    pl.col(column).is_null().sum().alias(f"null_{column}")
                    for column in ctr_columns
                ],
                *[
                    pl.col(column).mean().alias(f"coverage_{column}")
                    for column in history_columns
                ],
            )
            .collect()
            .with_columns(pl.lit(split).alias("split"))
        )
        rows = int(summary["rows"][0])
        observed_total += rows
        if rows != expected_rows:
            raise AssertionError(
                f"{split}: expected {expected_rows:,} rows, observed {rows:,}"
            )
        if int(summary["unique_impressions"][0]) != rows:
            raise AssertionError(f"{split}: impression_id is not unique")

        invalid_ctr = (
            scan.select(
                pl.any_horizontal(
                    *[
                        pl.col(column).is_null()
                        | (pl.col(column) < 0.0)
                        | (pl.col(column) > 1.0)
                        for column in ctr_columns
                    ]
                ).sum()
            )
            .collect()
            .item()
        )
        if invalid_ctr:
            raise AssertionError(f"{split}: found {invalid_ctr:,} invalid CTR values")

        first_date = summary["min_date"][0]
        if split == "train":
            first_day_nonzero = (
                scan.filter(pl.col(DATE_COLUMN) == first_date)
                .select(
                    pl.any_horizontal(
                        *[pl.col(column) != 0 for column in count_columns]
                    ).sum()
                )
                .collect()
                .item()
            )
            if first_day_nonzero:
                raise AssertionError(
                    "First training date contains nonzero past feedback; point-in-time rule failed"
                )
        summaries.append(summary)

    if observed_total != expected_total:
        raise AssertionError("Total output row count does not match source data")
    return pl.concat(summaries).select("split", pl.exclude("split"))


def build_daily_coverage() -> pl.DataFrame:
    tables = []
    for split in SPLITS:
        history_columns = [
            f"{prefix}_has_past_feedback" for prefix in ENTITY_COLUMNS
        ]
        table = (
            pl.scan_parquet(OUTPUT_DIR / f"{split}.parquet")
            .group_by(DATE_COLUMN)
            .agg(
                pl.len().alias("rows"),
                *[
                    pl.col(column).mean().alias(column.replace("has_", "coverage_"))
                    for column in history_columns
                ],
            )
            .collect()
            .with_columns(pl.lit(split).alias("split"))
        )
        tables.append(table)
    return pl.concat(tables).sort(DATE_COLUMN).select("split", pl.exclude("split"))


def main() -> None:
    args = parse_args()
    validate_args(args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOOKUP_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("BUILDING POINT-IN-TIME PAST CTR FEATURES")
    print("=" * 78)
    print(f"Smoothing strength: {args.smoothing_strength:g}")
    print(f"Initial prior:      {args.initial_prior:g}")
    print("Rule:               feature date < impression date (Beijing time)")

    source = all_impressions()
    print("\nBuilding global past-CTR prior by date...")
    global_lookup = build_global_lookup(source, args.initial_prior)
    global_lookup.write_parquet(LOOKUP_DIR / "global_daily_lookup.parquet", compression="zstd")
    print(global_lookup)

    for prefix, entity_column in ENTITY_COLUMNS.items():
        print(f"\nBuilding {prefix} lookup from {entity_column}...")
        lookup = build_entity_lookup(
            source,
            global_lookup,
            prefix,
            entity_column,
            args.smoothing_strength,
        )
        lookup.write_parquet(LOOKUP_DIR / f"{prefix}_daily_lookup.parquet", compression="zstd")
        print(f"  lookup rows: {lookup.height:,}")
        del lookup

    print("\nJoining F3 features to impression IDs...")
    for split in SPLITS:
        write_split(split, global_lookup)

    print("\n" + "=" * 78)
    print("PAST CTR VALIDATION")
    print("=" * 78)
    summary = validate_outputs()
    coverage = build_daily_coverage()
    print(summary)
    print("\nDaily feedback coverage:")
    print(coverage)

    summary.write_csv(RESULT_DIR / "split_summary.csv")
    coverage.write_csv(RESULT_DIR / "daily_coverage.csv")
    manifest = {
        "grain": "completed Beijing dates before each impression date",
        "source_splits": list(SPLITS),
        "entities": ENTITY_COLUMNS,
        "smoothing_strength": args.smoothing_strength,
        "initial_prior": args.initial_prior,
        "first_training_date_behavior": "all entity histories empty; fixed initial prior only",
        "test_history": "includes all completed dates before test, including validation date",
        "output_directory": str(OUTPUT_DIR),
    }
    (RESULT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nAll point-in-time validation checks passed.")
    print(f"Features: {OUTPUT_DIR}")
    print(f"Reports:  {RESULT_DIR}")


if __name__ == "__main__":
    main()
