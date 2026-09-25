from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl


RAW_PATH = Path("data/raw/behavior_log.csv")
OUTPUT_DIR = Path("data/processed/behavior")
OUTPUT_PATH = OUTPUT_DIR / "user_daily_behavior.parquet"

BEIJING_TZ = ZoneInfo("Asia/Shanghai")

# 官方行为日志的有效北京时间范围：
# 2017-04-22 00:00:00 <= behavior_time < 2017-05-14 00:00:00
VALID_START_DATE = date(2017, 4, 22)
VALID_END_EXCLUSIVE_DATE = date(2017, 5, 14)

EXPECTED_DATES = 22

BEHAVIOR_SCHEMA = {
    "user": pl.Int64,
    "time_stamp": pl.Int64,
    "btag": pl.String,
    "cate": pl.Int64,
    "brand": pl.Int64,
}

COUNT_COLUMNS = [
    "pv_count",
    "cart_count",
    "fav_count",
    "buy_count",
]


def beijing_midnight_epoch(day: date) -> int:
    """Convert Beijing midnight on a given date to Unix timestamp."""
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


def build_daily_behavior() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start_timestamp = beijing_midnight_epoch(VALID_START_DATE)
    end_timestamp = beijing_midnight_epoch(VALID_END_EXCLUSIVE_DATE)

    print("=" * 70)
    print("BUILDING DAILY USER BEHAVIOR")
    print("=" * 70)
    print(f"Input:  {RAW_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(
        "Valid Beijing window: "
        f"{VALID_START_DATE} <= date < {VALID_END_EXCLUSIVE_DATE}"
    )
    print(f"Unix timestamp window: {start_timestamp} <= ts < {end_timestamp}")
    print()
    print("Scanning and aggregating the 22 GB behavior log.")
    print("This can take substantially longer than the validation scripts.")

    behavior = (
        pl.scan_csv(
            RAW_PATH,
            schema=BEHAVIOR_SCHEMA,
            null_values=[""],
        )
        .filter(
            (pl.col("time_stamp") >= start_timestamp)
            & (pl.col("time_stamp") < end_timestamp)
        )
        .select(
            pl.col("user").alias("user_raw"),
            pl.col("time_stamp"),
            pl.col("btag"),
        )
        .with_columns(
            pl.from_epoch(
                pl.col("time_stamp"),
                time_unit="s",
            )
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("Asia/Shanghai")
            .dt.date()
            .alias("event_date_bj")
        )
    )

    daily_behavior = (
        behavior.group_by(["user_raw", "event_date_bj"])
        .agg(
            pl.len().cast(pl.UInt32).alias("event_count"),
            (pl.col("btag") == "pv")
            .sum()
            .cast(pl.UInt32)
            .alias("pv_count"),
            (pl.col("btag") == "cart")
            .sum()
            .cast(pl.UInt32)
            .alias("cart_count"),
            (pl.col("btag") == "fav")
            .sum()
            .cast(pl.UInt32)
            .alias("fav_count"),
            (pl.col("btag") == "buy")
            .sum()
            .cast(pl.UInt32)
            .alias("buy_count"),
            pl.col("time_stamp")
            .max()
            .alias("last_event_timestamp"),
            pl.col("time_stamp")
            .filter(pl.col("btag") == "buy")
            .max()
            .alias("last_buy_timestamp"),
        )
    )

    daily_behavior.sink_parquet(
        OUTPUT_PATH,
        compression="zstd",
        statistics=True,
    )

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"File size: {size_mb:,.1f} MB")


def validate_daily_behavior() -> None:
    print()
    print("=" * 70)
    print("DAILY BEHAVIOR VALIDATION")
    print("=" * 70)

    daily = pl.scan_parquet(OUTPUT_PATH)

    summary = (
        daily.select(
            pl.len().alias("daily_rows"),
            pl.col("user_raw").n_unique().alias("unique_users"),
            pl.col("event_date_bj").n_unique().alias("unique_dates"),
            pl.col("event_date_bj").min().alias("min_date"),
            pl.col("event_date_bj").max().alias("max_date"),
            pl.col("event_count").sum().alias("total_events"),
            pl.col("pv_count").sum().alias("total_pv"),
            pl.col("cart_count").sum().alias("total_cart"),
            pl.col("fav_count").sum().alias("total_fav"),
            pl.col("buy_count").sum().alias("total_buy"),
        )
        .collect()
    )

    print(summary)

    stats = summary.row(0, named=True)

    counted_events = (
        int(stats["total_pv"])
        + int(stats["total_cart"])
        + int(stats["total_fav"])
        + int(stats["total_buy"])
    )

    if stats["min_date"] != VALID_START_DATE:
        raise ValueError(
            f"Unexpected minimum date: {stats['min_date']}"
        )

    expected_max_date = date(2017, 5, 13)
    if stats["max_date"] != expected_max_date:
        raise ValueError(
            f"Unexpected maximum date: {stats['max_date']}"
        )

    if int(stats["unique_dates"]) != EXPECTED_DATES:
        raise ValueError(
            "Unexpected number of dates: "
            f"{stats['unique_dates']} instead of {EXPECTED_DATES}"
        )

    if int(stats["total_events"]) != counted_events:
        raise ValueError(
            "Behavior counts do not add up: "
            f"event_count={stats['total_events']}, "
            f"sum of btag counts={counted_events}"
        )

    bad_group_count = (
        daily.filter(
            pl.col("event_count")
            != pl.sum_horizontal(
                [pl.col(column) for column in COUNT_COLUMNS]
            )
        )
        .select(pl.len())
        .collect()
        .item()
    )

    if bad_group_count != 0:
        raise ValueError(
            f"Found {bad_group_count} daily rows with inconsistent counts"
        )

    daily_totals = (
        daily.group_by("event_date_bj")
        .agg(
            pl.col("event_count").sum().alias("events"),
            pl.col("pv_count").sum().alias("pv"),
            pl.col("cart_count").sum().alias("cart"),
            pl.col("fav_count").sum().alias("fav"),
            pl.col("buy_count").sum().alias("buy"),
        )
        .sort("event_date_bj")
        .collect()
    )

    print("\nDaily event totals:")
    print(daily_totals)

    print("\nAll daily-behavior validation checks passed.")


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw file: {RAW_PATH}")

    build_daily_behavior()
    validate_daily_behavior()

    print()
    print("Done.")
    print(f"Output is in: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()