from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl


DAILY_BEHAVIOR_PATH = Path(
    "data/processed/behavior/user_daily_behavior.parquet"
)
BASE_DIR = Path("data/processed/base")
LOOKUP_DIR = Path("data/processed/behavior/window_features")
OUTPUT_DIR = Path("data/processed/features")

BEIJING_TZ = ZoneInfo("Asia/Shanghai")

TARGET_DATES = [
    date(2017, 5, day)
    for day in range(6, 14)
]

WINDOWS = (1, 3, 7, 14)

METRICS = (
    "event_count",
    "pv_count",
    "cart_count",
    "fav_count",
    "buy_count",
)

EXPECTED_ROWS = {
    "train": 20_015_245,
    "val": 3_234_051,
    "test": 3_308_665,
}

COUNT_FEATURE_COLUMNS = [
    f"{metric}_{window}d"
    for window in WINDOWS
    for metric in METRICS
]


def beijing_midnight_epoch(day: date) -> int:
    value = datetime(
        day.year,
        day.month,
        day.day,
        0,
        0,
        0,
        tzinfo=BEIJING_TZ,
    )
    return int(value.timestamp())


def build_lookup_for_date(target_date: date) -> Path:
    """
    Build one row per user for one target impression date.

    Only behavior dates strictly earlier than target_date are used.
    """
    maximum_window = max(WINDOWS)
    history_start = target_date - timedelta(days=maximum_window)

    output_path = (
        LOOKUP_DIR
        / f"user_behavior_{target_date.isoformat()}.parquet"
    )

    print(
        f"Building behavior lookup for {target_date}: "
        f"{history_start} <= behavior date < {target_date}"
    )

    daily = (
        pl.scan_parquet(DAILY_BEHAVIOR_PATH)
        .filter(
            (pl.col("event_date_bj") >= history_start)
            & (pl.col("event_date_bj") < target_date)
        )
    )

    aggregate_expressions = []

    for window in WINDOWS:
        window_start = target_date - timedelta(days=window)

        for metric in METRICS:
            aggregate_expressions.append(
                pl.col(metric)
                .filter(pl.col("event_date_bj") >= window_start)
                .sum()
                .cast(pl.UInt32)
                .alias(f"{metric}_{window}d")
            )

    aggregate_expressions.extend(
        [
            pl.col("last_event_timestamp")
            .max()
            .alias("last_event_timestamp_14d"),
            pl.col("last_buy_timestamp")
            .max()
            .alias("last_buy_timestamp_14d"),
        ]
    )

    target_timestamp = beijing_midnight_epoch(target_date)

    features = (
        daily.group_by("user_raw")
        .agg(aggregate_expressions)
        .with_columns(
            pl.lit(target_date).alias("event_date_bj"),
            pl.lit(1)
            .cast(pl.Int8)
            .alias("has_behavior_history_14d"),
        )
        .with_columns(
            (
                (
                    pl.lit(target_timestamp)
                    - pl.col("last_event_timestamp_14d")
                )
                / 86_400
            )
            .cast(pl.Float32)
            .alias("days_since_last_event_14d"),
            (
                (
                    pl.lit(target_timestamp)
                    - pl.col("last_buy_timestamp_14d")
                )
                / 86_400
            )
            .cast(pl.Float32)
            .alias("days_since_last_buy_14d"),
            (
                pl.col("buy_count_14d") > 0
            )
            .cast(pl.Int8)
            .alias("has_buy_history_14d"),
        )
        .drop(
            [
                "last_event_timestamp_14d",
                "last_buy_timestamp_14d",
            ]
        )
        .select(
            "user_raw",
            "event_date_bj",
            "has_behavior_history_14d",
            "has_buy_history_14d",
            *COUNT_FEATURE_COLUMNS,
            "days_since_last_event_14d",
            "days_since_last_buy_14d",
        )
    )

    features.sink_parquet(
        output_path,
        compression="zstd",
        statistics=True,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {output_path.name}: {size_mb:,.1f} MB")

    return output_path


def build_all_lookup_tables() -> None:
    print("=" * 72)
    print("BUILDING LEAKAGE-FREE BEHAVIOR WINDOW FEATURES")
    print("=" * 72)

    LOOKUP_DIR.mkdir(parents=True, exist_ok=True)

    for target_date in TARGET_DATES:
        build_lookup_for_date(target_date)


def load_all_lookup_tables() -> pl.LazyFrame:
    pattern = str(LOOKUP_DIR / "user_behavior_*.parquet")
    return pl.scan_parquet(pattern)


def join_features_to_split(
    split_name: str,
    behavior_features: pl.LazyFrame,
) -> None:
    input_path = BASE_DIR / f"{split_name}.parquet"
    output_path = OUTPUT_DIR / f"{split_name}.parquet"

    print(f"\nJoining behavior features into {split_name}...")

    base = pl.scan_parquet(input_path)

    enriched = (
        base.join(
            behavior_features,
            on=["user_raw", "event_date_bj"],
            how="left",
        )
        .with_columns(
            [
                pl.col(column)
                .fill_null(0)
                .cast(pl.UInt32)
                for column in COUNT_FEATURE_COLUMNS
            ]
            + [
                pl.col("has_behavior_history_14d")
                .fill_null(0)
                .cast(pl.Int8),
                pl.col("has_buy_history_14d")
                .fill_null(0)
                .cast(pl.Int8),
            ]
        )
    )

    enriched.sink_parquet(
        output_path,
        compression="zstd",
        statistics=True,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {output_path}: {size_mb:,.1f} MB")


def join_all_splits() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    behavior_features = load_all_lookup_tables()

    for split_name in ("train", "val", "test"):
        join_features_to_split(
            split_name,
            behavior_features,
        )


def validate_split(split_name: str) -> dict:
    path = OUTPUT_DIR / f"{split_name}.parquet"
    data = pl.scan_parquet(path)

    summary = (
        data.select(
            pl.lit(split_name).alias("split"),
            pl.len().alias("rows"),
            pl.col("impression_id")
            .n_unique()
            .alias("unique_impressions"),
            pl.col("has_behavior_history_14d")
            .mean()
            .alias("behavior_coverage"),
            pl.col("has_buy_history_14d")
            .mean()
            .alias("buy_history_coverage"),
            pl.col("event_count_1d")
            .mean()
            .alias("mean_events_1d"),
            pl.col("event_count_7d")
            .mean()
            .alias("mean_events_7d"),
            pl.col("event_count_14d")
            .mean()
            .alias("mean_events_14d"),
            pl.col("days_since_last_event_14d")
            .min()
            .alias("min_event_recency"),
            pl.col("days_since_last_event_14d")
            .max()
            .alias("max_event_recency"),
        )
        .collect()
    )

    stats = summary.row(0, named=True)

    if int(stats["rows"]) != EXPECTED_ROWS[split_name]:
        raise ValueError(
            f"{split_name}: row count changed after join: "
            f"{stats['rows']} instead of "
            f"{EXPECTED_ROWS[split_name]}"
        )

    if int(stats["unique_impressions"]) != EXPECTED_ROWS[split_name]:
        raise ValueError(
            f"{split_name}: duplicate impression IDs detected"
        )

    invalid_conditions = []

    for window in WINDOWS:
        invalid_conditions.append(
            pl.col(f"event_count_{window}d")
            != pl.sum_horizontal(
                [
                    pl.col(f"pv_count_{window}d"),
                    pl.col(f"cart_count_{window}d"),
                    pl.col(f"fav_count_{window}d"),
                    pl.col(f"buy_count_{window}d"),
                ]
            )
        )

    for metric in METRICS:
        invalid_conditions.extend(
            [
                pl.col(f"{metric}_1d")
                > pl.col(f"{metric}_3d"),
                pl.col(f"{metric}_3d")
                > pl.col(f"{metric}_7d"),
                pl.col(f"{metric}_7d")
                > pl.col(f"{metric}_14d"),
            ]
        )

    invalid_conditions.extend(
        [
            pl.col("days_since_last_event_14d") < 0,
            pl.col("days_since_last_event_14d") > 14.0001,
            pl.col("days_since_last_buy_14d") < 0,
            pl.col("days_since_last_buy_14d") > 14.0001,
        ]
    )

    invalid_rows = (
        data.filter(
            pl.any_horizontal(invalid_conditions)
        )
        .select(pl.len())
        .collect()
        .item()
    )

    if invalid_rows != 0:
        raise ValueError(
            f"{split_name}: found {invalid_rows} "
            "rows with invalid behavior features"
        )

    return stats


def validate_outputs() -> None:
    print()
    print("=" * 72)
    print("BEHAVIOR FEATURE VALIDATION")
    print("=" * 72)

    summaries = []

    for split_name in ("train", "val", "test"):
        summaries.append(validate_split(split_name))

    print(pl.DataFrame(summaries))

    total_rows = sum(
        int(summary["rows"])
        for summary in summaries
    )

    expected_total = sum(EXPECTED_ROWS.values())

    if total_rows != expected_total:
        raise ValueError(
            f"Total row count is {total_rows}, "
            f"expected {expected_total}"
        )

    print(f"\nTotal rows:    {total_rows:,}")
    print(f"Expected rows: {expected_total:,}")
    print("All behavior-feature validation checks passed.")


def main() -> None:
    if not DAILY_BEHAVIOR_PATH.exists():
        raise FileNotFoundError(
            f"Missing daily behavior file: "
            f"{DAILY_BEHAVIOR_PATH}"
        )

    for split_name in ("train", "val", "test"):
        path = BASE_DIR / f"{split_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing base table: {path}"
            )

    build_all_lookup_tables()
    join_all_splits()
    validate_outputs()

    print()
    print("Done.")
    print(f"Final feature tables are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()