from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl


RAW_DIR = Path("data/raw")
BJ = ZoneInfo("Asia/Shanghai")


SCHEMAS = {
    "raw_sample.csv": {
        "user": pl.Int64,
        "time_stamp": pl.Int64,
        "adgroup_id": pl.Int64,
        "pid": pl.String,
        "nonclk": pl.Int8,
        "clk": pl.Int8,
    },
    "ad_feature.csv": {
        "adgroup_id": pl.Int64,
        "cate_id": pl.Int64,
        "campaign_id": pl.Int64,
        "customer": pl.Int64,
        "brand": pl.Float64,
        "price": pl.Float64,
    },
    "user_profile.csv": {
        "userid": pl.Int64,
        "cms_segid": pl.Int64,
        "cms_group_id": pl.Int64,
        "final_gender_code": pl.Int64,
        "age_level": pl.Int64,
        "pvalue_level": pl.Float64,
        "shopping_level": pl.Int64,
        "occupation": pl.Int64,
        "new_user_class_level ": pl.Float64,
    },
    "behavior_log.csv": {
        "user": pl.Int64,
        "time_stamp": pl.Int64,
        "btag": pl.String,
        "cate": pl.Int64,
        "brand": pl.Int64,
    },
}


def scan_csv(name: str) -> pl.LazyFrame:
    path = RAW_DIR / name

    if not path.exists():
        raise FileNotFoundError(path)

    return pl.scan_csv(
        path,
        schema=SCHEMAS[name],
        null_values=["", "NULL", "null"],
    )


def format_timestamp(timestamp: int) -> str:
    utc_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    bj_time = datetime.fromtimestamp(timestamp, tz=BJ)
    return f"UTC={utc_time.isoformat()} | Beijing={bj_time.isoformat()}"


def main() -> None:
    raw_sample = scan_csv("raw_sample.csv")
    ad_feature = scan_csv("ad_feature.csv")
    user_profile = (
        scan_csv("user_profile.csv")
        .rename({"new_user_class_level ": "new_user_class_level"})
    )
    behavior_log = scan_csv("behavior_log.csv")

    print("Validating raw_sample.csv...")
    raw_stats = raw_sample.select(
        pl.len().alias("rows"),
        pl.col("user").n_unique().alias("unique_users"),
        pl.col("adgroup_id").n_unique().alias("unique_ads"),
        pl.col("time_stamp").min().alias("min_timestamp"),
        pl.col("time_stamp").max().alias("max_timestamp"),
        (~pl.col("clk").is_in([0, 1])).sum().alias("invalid_clk"),
        (pl.col("clk") + pl.col("nonclk") != 1)
        .sum()
        .alias("invalid_label_pairs"),
        pl.col("clk").mean().alias("global_ctr"),
    ).collect(engine="streaming")

    print(raw_stats)

    min_ts = raw_stats["min_timestamp"][0]
    max_ts = raw_stats["max_timestamp"][0]

    print("raw_sample time range:")
    print("  min:", format_timestamp(min_ts))
    print("  max:", format_timestamp(max_ts))

    print("\nValidating ad_feature.csv...")
    ad_stats = ad_feature.select(
        pl.len().alias("rows"),
        pl.col("adgroup_id").n_unique().alias("unique_ads"),
        pl.col("adgroup_id").is_null().sum().alias("null_adgroup_id"),
        pl.col("price").is_null().sum().alias("null_price"),
        (pl.col("price") < 0).sum().alias("negative_price"),
        pl.col("price").max().alias("max_price"),
    ).collect(engine="streaming")

    print(ad_stats)

    print("\nValidating user_profile.csv...")
    user_stats = user_profile.select(
        pl.len().alias("rows"),
        pl.col("userid").n_unique().alias("unique_users"),
        pl.col("userid").is_null().sum().alias("null_userid"),
        pl.col("pvalue_level").is_null().sum().alias("null_pvalue"),
        pl.col("new_user_class_level")
        .is_null()
        .sum()
        .alias("null_new_user_class"),
    ).collect(engine="streaming")

    print(user_stats)

    print("\nValidating behavior_log.csv...")
    print("This scans the 22 GB behavior file and may take several minutes.")

    behavior_stats = behavior_log.select(
        pl.len().alias("rows"),
        pl.col("user").n_unique().alias("unique_users"),
        pl.col("time_stamp").min().alias("min_timestamp"),
        pl.col("time_stamp").max().alias("max_timestamp"),
        (~pl.col("btag").is_in(["pv", "cart", "fav", "buy"]))
        .sum()
        .alias("invalid_btag"),
        pl.col("cate").is_null().sum().alias("null_cate"),
    ).collect(engine="streaming")

    print(behavior_stats)

    behavior_min = behavior_stats["min_timestamp"][0]
    behavior_max = behavior_stats["max_timestamp"][0]

    print("behavior_log time range:")
    print("  min:", format_timestamp(behavior_min))
    print("  max:", format_timestamp(behavior_max))

    print("\nRaw-data validation completed successfully.")


if __name__ == "__main__":
    main()