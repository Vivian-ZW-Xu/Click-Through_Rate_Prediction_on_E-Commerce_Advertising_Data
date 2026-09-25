from datetime import date
from pathlib import Path

import polars as pl


RAW_DIR = Path("data/raw")


AD_SCHEMA = {
    "adgroup_id": pl.Int64,
    "cate_id": pl.Int64,
    "campaign_id": pl.Int64,
    "customer": pl.Int64,
    "brand": pl.Float64,
    "price": pl.Float64,
}

USER_SCHEMA = {
    "userid": pl.Int64,
    "cms_segid": pl.Int64,
    "cms_group_id": pl.Int64,
    "final_gender_code": pl.Int64,
    "age_level": pl.Int64,
    "pvalue_level": pl.Float64,
    "shopping_level": pl.Int64,
    "occupation": pl.Int64,
    "new_user_class_level ": pl.Float64,
}

BEHAVIOR_SCHEMA = {
    "user": pl.Int64,
    "time_stamp": pl.Int64,
    "btag": pl.String,
    "cate": pl.Int64,
    "brand": pl.Int64,
}


def profile_ad_features() -> None:
    print("=" * 70)
    print("AD FEATURE PROFILE")
    print("=" * 70)

    ads = pl.scan_csv(
        RAW_DIR / "ad_feature.csv",
        schema=AD_SCHEMA,
        null_values=["", "NULL", "null"],
    )

    price_stats = ads.select(
        pl.len().alias("rows"),
        pl.col("price").min().alias("min"),
        pl.col("price").median().alias("median"),
        pl.col("price").quantile(0.90).alias("p90"),
        pl.col("price").quantile(0.95).alias("p95"),
        pl.col("price").quantile(0.99).alias("p99"),
        pl.col("price").quantile(0.999).alias("p999"),
        pl.col("price").quantile(0.9999).alias("p9999"),
        pl.col("price").max().alias("max"),
        (pl.col("price") > 500_000).sum().alias("price_gt_500k"),
        (pl.col("price") == 99_999_999).sum().alias("price_eq_99999999"),
        pl.col("brand").is_null().sum().alias("null_brand"),
        pl.col("brand").n_unique().alias("unique_brand"),
    ).collect(engine="streaming")

    print("\nPrice and brand statistics:")
    print(price_stats)

    print("\nTop 20 most expensive ads:")
    top_prices = (
        ads.select("adgroup_id", "cate_id", "brand", "price")
        .sort("price", descending=True)
        .head(20)
        .collect(engine="streaming")
    )
    print(top_prices)


def profile_user_features() -> None:
    print("\n" + "=" * 70)
    print("USER PROFILE")
    print("=" * 70)

    users = (
        pl.read_csv(
            RAW_DIR / "user_profile.csv",
            schema=USER_SCHEMA,
            null_values=["", "NULL", "null"],
        )
        .rename({"new_user_class_level ": "new_user_class_level"})
    )

    feature_columns = [
        "cms_segid",
        "cms_group_id",
        "final_gender_code",
        "age_level",
        "pvalue_level",
        "shopping_level",
        "occupation",
        "new_user_class_level",
    ]

    summary = users.select([
        pl.len().alias("rows"),
        *[
            pl.col(column).null_count().alias(f"{column}_null")
            for column in feature_columns
        ],
    ])

    print("\nNull counts:")
    print(summary)

    for column in feature_columns:
        n_unique = users[column].n_unique()
        n_null = users[column].null_count()

        print("\n" + "-" * 50)
        print(f"Feature: {column}")
        print(f"Unique values: {n_unique}")
        print(f"Null values:   {n_null}")

        value_counts = (
            users.group_by(column)
            .len()
            .sort("len", descending=True)
            .head(20)
        )

        print("Top value counts:")
        print(value_counts)


def profile_behavior_dates() -> None:
    print("\n" + "=" * 70)
    print("BEHAVIOR LOG DATE PROFILE")
    print("=" * 70)
    print("Scanning 22 GB behavior_log.csv. This may take several minutes.")

    behavior = pl.scan_csv(
        RAW_DIR / "behavior_log.csv",
        schema=BEHAVIOR_SCHEMA,
        null_values=["", "NULL", "null"],
    )

    behavior_with_date = behavior.with_columns(
        (
            pl.from_epoch("time_stamp", time_unit="s")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("Asia/Shanghai")
            .dt.date()
        ).alias("beijing_date")
    )

    daily_counts = (
        behavior_with_date
        .group_by("beijing_date")
        .agg(pl.len().alias("events"))
        .sort("beijing_date")
        .collect(engine="streaming")
    )

    report_dir = Path("results")
    report_dir.mkdir(parents=True, exist_ok=True)

    daily_counts.write_csv(
        report_dir / "behavior_daily_counts.csv"
    )

    print("\nHigh-volume behavior days:")
    print(
        daily_counts.filter(
            pl.col("events") >= 1_000_000
        )
    )

    # This is deliberately a broad window.
    # We use it to discover the true continuous behavior-log period.
    broad_start = date(2017, 4, 1)
    broad_end = date(2017, 5, 14)  # exclusive

    in_broad_window = daily_counts.filter(
        (pl.col("beijing_date") >= broad_start)
        & (pl.col("beijing_date") < broad_end)
    )

    outside_window = daily_counts.filter(
        (pl.col("beijing_date") < broad_start)
        | (pl.col("beijing_date") >= broad_end)
    )

    outside_events = outside_window["events"].sum()
    total_events = daily_counts["events"].sum()

    print("\nDaily behavior counts within 2017-04-01 to 2017-05-13:")
    print(in_broad_window)

    print("\nTimestamp anomaly summary:")
    print(f"Total events:                 {total_events:,}")
    print(f"Events outside broad window: {outside_events:,}")
    print(f"Outside-window dates:         {outside_window.height:,}")

    if outside_window.height > 0:
        print("\nTop 20 anomalous dates by event count:")
        print(
            outside_window
            .sort("events", descending=True)
            .head(20)
        )


def main() -> None:
    profile_ad_features()
    profile_user_features()
    profile_behavior_dates()

    print("\n" + "=" * 70)
    print("RAW DATA PROFILING FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()