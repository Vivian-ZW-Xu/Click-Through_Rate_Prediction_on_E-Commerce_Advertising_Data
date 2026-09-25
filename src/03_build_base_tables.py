from datetime import date
from pathlib import Path

import polars as pl


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed/base")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_START = date(2017, 5, 6)
VAL_DATE = date(2017, 5, 12)
TEST_DATE = date(2017, 5, 13)

EXPECTED_TOTAL_ROWS = 26_557_961
PRICE_SENTINEL = 99_999_999.0


RAW_SAMPLE_SCHEMA = {
    "user": pl.Int64,
    "time_stamp": pl.Int64,
    "adgroup_id": pl.Int64,
    "pid": pl.String,
    "nonclk": pl.Int8,
    "clk": pl.Int8,
}

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


def load_ad_features() -> pl.DataFrame:
    """Read and minimally clean the static ad table."""

    ads = (
        pl.read_csv(
            RAW_DIR / "ad_feature.csv",
            schema=AD_SCHEMA,
            null_values=["", "NULL", "null"],
        )
        .rename({
            "adgroup_id": "adgroup_id_raw",
            "cate_id": "cate_id_raw",
            "campaign_id": "campaign_id_raw",
            "customer": "customer_raw",
            "brand": "brand_raw",
            "price": "price_raw",
        })
        .with_columns(
            # Brand IDs are categorical integers, although the CSV was read
            # as Float64 because it contains null values.
            pl.col("brand_raw").cast(pl.Int64),

            (pl.col("price_raw") == PRICE_SENTINEL)
            .cast(pl.Int8)
            .alias("price_is_sentinel"),

            pl.col("brand_raw")
            .is_null()
            .cast(pl.Int8)
            .alias("brand_is_missing"),
        )
        .with_columns(
            # Preserve price_raw for auditing. Only the known sentinel is
            # converted to null; no advertisements are deleted.
            pl.when(pl.col("price_is_sentinel") == 1)
            .then(None)
            .otherwise(pl.col("price_raw"))
            .alias("price_clean")
        )
        .with_columns(
            (pl.col("price_clean") + 1)
            .log()
            .cast(pl.Float32)
            .alias("log_price")
        )
    )

    if ads["adgroup_id_raw"].n_unique() != ads.height:
        raise ValueError("ad_feature contains duplicate adgroup_id values")

    print(f"Loaded ad features: {ads.height:,} unique advertisements")
    return ads


def load_user_profiles() -> pl.DataFrame:
    """Read the user profile table while preserving null values."""

    users = (
        pl.read_csv(
            RAW_DIR / "user_profile.csv",
            schema=USER_SCHEMA,
            null_values=["", "NULL", "null"],
        )
        .rename({
            "userid": "user_raw",
            "new_user_class_level ": "new_user_class_level",
        })
        .with_columns(
            # profile_present is only a temporary join marker.
            pl.lit(1, dtype=pl.Int8).alias("profile_present"),

            # These are categorical levels, not continuous measurements.
            pl.col("cms_segid").cast(pl.Int16),
            pl.col("cms_group_id").cast(pl.Int16),
            pl.col("final_gender_code").cast(pl.Int8),
            pl.col("age_level").cast(pl.Int8),
            pl.col("pvalue_level").cast(pl.Int8),
            pl.col("shopping_level").cast(pl.Int8),
            pl.col("occupation").cast(pl.Int8),
            pl.col("new_user_class_level").cast(pl.Int8),
        )
    )

    if users["user_raw"].n_unique() != users.height:
        raise ValueError("user_profile contains duplicate user IDs")

    print(f"Loaded user profiles: {users.height:,} unique users")
    return users


def build_base_lazy(
    ads: pl.DataFrame,
    users: pl.DataFrame,
) -> pl.LazyFrame:
    """Create the joined base table without model-specific encoding."""

    impressions = (
        pl.scan_csv(
            RAW_DIR / "raw_sample.csv",
            schema=RAW_SAMPLE_SCHEMA,
            null_values=["", "NULL", "null"],
        )
        .with_row_index("impression_id")
        .drop("nonclk")
        .rename({
            "user": "user_raw",
            "adgroup_id": "adgroup_id_raw",
            "pid": "pid_raw",
        })
    )

    base = (
        impressions
        .join(
            ads.lazy(),
            on="adgroup_id_raw",
            how="left",
        )
        .join(
            users.lazy(),
            on="user_raw",
            how="left",
        )
        .with_columns(
            pl.col("profile_present")
            .fill_null(0)
            .cast(pl.Int8)
            .alias("has_profile"),

            pl.from_epoch("time_stamp", time_unit="s")
            .dt.replace_time_zone("UTC")
            .alias("event_time_utc"),
        )
        .drop("profile_present")
        .with_columns(
            pl.col("event_time_utc")
            .dt.convert_time_zone("Asia/Shanghai")
            .alias("event_time_bj")
        )
        .with_columns(
            pl.col("event_time_bj")
            .dt.date()
            .alias("event_date_bj"),

            pl.col("event_time_bj")
            .dt.hour()
            .cast(pl.Int8)
            .alias("hour"),

            pl.col("event_time_bj")
            .dt.weekday()
            .cast(pl.Int8)
            .alias("day_of_week"),
        )
        .with_columns(
            pl.when(
                (pl.col("event_date_bj") >= DATA_START)
                & (pl.col("event_date_bj") < VAL_DATE)
            )
            .then(pl.lit("train"))
            .when(pl.col("event_date_bj") == VAL_DATE)
            .then(pl.lit("val"))
            .when(pl.col("event_date_bj") == TEST_DATE)
            .then(pl.lit("test"))
            .otherwise(pl.lit("out_of_range"))
            .alias("split")
        )
    )

    return base


def write_splits(base: pl.LazyFrame) -> None:
    """Materialize each time split as an independent parquet file."""

    for split_name in ["train", "val", "test"]:
        output_path = OUTPUT_DIR / f"{split_name}.parquet"

        print(f"\nWriting {split_name} -> {output_path}")

        (
            base
            .filter(pl.col("split") == split_name)
            .sink_parquet(
                output_path,
                compression="zstd",
                statistics=True,
            )
        )

        size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"Saved {output_path.name}: {size_mb:.1f} MB")


def validate_outputs() -> None:
    """Check joins, row conservation, time boundaries and CTR."""

    summaries = []

    for split_name in ["train", "val", "test"]:
        path = OUTPUT_DIR / f"{split_name}.parquet"

        summary = (
            pl.scan_parquet(path)
            .select(
                pl.lit(split_name).alias("split"),
                pl.len().alias("rows"),
                pl.col("clk").mean().alias("ctr"),
                pl.col("event_time_bj").min().alias("min_time"),
                pl.col("event_time_bj").max().alias("max_time"),

                pl.col("cate_id_raw")
                .is_null()
                .sum()
                .alias("missing_ad_join"),

                (pl.col("has_profile") == 0)
                .sum()
                .alias("missing_profile_rows"),

                pl.col("price_is_sentinel")
                .sum()
                .alias("sentinel_price_rows"),

                pl.col("brand_is_missing")
                .sum()
                .alias("missing_brand_rows"),
            )
            .collect()
        )

        summaries.append(summary)

    result = pl.concat(summaries)

    print("\n" + "=" * 80)
    print("BASE TABLE VALIDATION")
    print("=" * 80)
    print(result)

    total_rows = result["rows"].sum()
    missing_ad_rows = result["missing_ad_join"].sum()

    print(f"\nTotal output rows: {total_rows:,}")
    print(f"Expected rows:     {EXPECTED_TOTAL_ROWS:,}")
    print(f"Missing ad joins:  {missing_ad_rows:,}")

    if total_rows != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Row count mismatch: expected {EXPECTED_TOTAL_ROWS:,}, "
            f"got {total_rows:,}"
        )

    if missing_ad_rows != 0:
        raise ValueError(
            f"{missing_ad_rows:,} impression rows failed to join ad features"
        )

    for split_name in ["train", "val", "test"]:
        path = OUTPUT_DIR / f"{split_name}.parquet"

        bad_split_rows = (
            pl.scan_parquet(path)
            .filter(pl.col("split") != split_name)
            .select(pl.len())
            .collect()
            .item()
        )

        if bad_split_rows != 0:
            raise ValueError(
                f"{split_name} contains {bad_split_rows:,} incorrectly "
                "assigned rows"
            )

    print("\nAll base-table validation checks passed.")


def main() -> None:
    print("Building leakage-free base tables...\n")

    ads = load_ad_features()
    users = load_user_profiles()
    base = build_base_lazy(ads, users)

    write_splits(base)
    validate_outputs()

    print("\nDone.")
    print(f"Outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()