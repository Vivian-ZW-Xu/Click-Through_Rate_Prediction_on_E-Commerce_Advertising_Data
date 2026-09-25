import json
from pathlib import Path

import polars as pl


INPUT_DIR = Path("data/processed/features")
OUTPUT_DIR = Path("results/feature_profile")

SPLITS = ("train", "val", "test")

CATEGORICAL_FEATURES = [
    "user_raw",
    "adgroup_id_raw",
    "pid_raw",
    "cate_id_raw",
    "campaign_id_raw",
    "customer_raw",
    "brand_raw",
    "cms_segid",
    "cms_group_id",
    "final_gender_code",
    "age_level",
    "pvalue_level",
    "shopping_level",
    "occupation",
    "new_user_class_level",
    "hour",
    "day_of_week",
]

BINARY_FEATURES = [
    "price_is_sentinel",
    "brand_is_missing",
    "has_profile",
    "has_behavior_history_14d",
    "has_buy_history_14d",
]

BEHAVIOR_COUNT_FEATURES = [
    f"{metric}_{window}d"
    for window in (1, 3, 7, 14)
    for metric in (
        "event_count",
        "pv_count",
        "cart_count",
        "fav_count",
        "buy_count",
    )
]

NUMERIC_FEATURES = [
    "log_price",
    *BEHAVIOR_COUNT_FEATURES,
    "days_since_last_event_14d",
    "days_since_last_buy_14d",
]

METADATA_COLUMNS = [
    "impression_id",
    "time_stamp",
    "event_time_utc",
    "event_time_bj",
    "event_date_bj",
    "split",
]

EXCLUDED_REDUNDANT_COLUMNS = [
    "price_raw",
    "price_clean",
]

LABEL_COLUMN = "clk"


def load_split(split_name: str) -> pl.LazyFrame:
    path = INPUT_DIR / f"{split_name}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    return pl.scan_parquet(path)


def validate_schema() -> dict:
    print("=" * 76)
    print("MODEL FEATURE SCHEMA")
    print("=" * 76)

    schemas = {
        split_name: load_split(split_name).collect_schema()
        for split_name in SPLITS
    }

    train_schema = schemas["train"]
    train_columns = train_schema.names()

    for split_name in ("val", "test"):
        if schemas[split_name] != train_schema:
            raise ValueError(
                f"{split_name} schema does not match train schema"
            )

    assigned_columns = set(
        CATEGORICAL_FEATURES
        + BINARY_FEATURES
        + NUMERIC_FEATURES
        + METADATA_COLUMNS
        + EXCLUDED_REDUNDANT_COLUMNS
        + [LABEL_COLUMN]
    )

    actual_columns = set(train_columns)

    missing_columns = sorted(assigned_columns - actual_columns)
    unassigned_columns = sorted(actual_columns - assigned_columns)

    if missing_columns:
        raise ValueError(
            f"Expected columns are missing: {missing_columns}"
        )

    if unassigned_columns:
        raise ValueError(
            f"Columns have not been assigned a role: "
            f"{unassigned_columns}"
        )

    suspicious_columns = [
        column
        for column in train_columns
        if (
            "ctr" in column.lower()
            or "click_rate" in column.lower()
            or "target_mean" in column.lower()
            or "label_mean" in column.lower()
        )
    ]

    if suspicious_columns:
        raise ValueError(
            "Potential label-derived columns detected: "
            f"{suspicious_columns}"
        )

    input_features = (
        CATEGORICAL_FEATURES
        + BINARY_FEATURES
        + NUMERIC_FEATURES
    )

    if LABEL_COLUMN in input_features:
        raise ValueError("clk was accidentally included as an input")

    print(f"Total columns:       {len(train_columns)}")
    print(f"Categorical inputs:  {len(CATEGORICAL_FEATURES)}")
    print(f"Binary inputs:       {len(BINARY_FEATURES)}")
    print(f"Numeric inputs:      {len(NUMERIC_FEATURES)}")
    print(f"Metadata columns:    {len(METADATA_COLUMNS)}")
    print(f"Excluded columns:    {len(EXCLUDED_REDUNDANT_COLUMNS)}")
    print(f"Label:               {LABEL_COLUMN}")
    print("No suspicious CTR or target-mean columns found.")

    return train_schema


def build_split_summary() -> pl.DataFrame:
    rows = []

    for split_name in SPLITS:
        summary = (
            load_split(split_name)
            .select(
                pl.len().alias("rows"),
                pl.col(LABEL_COLUMN).sum().alias("clicks"),
                pl.col(LABEL_COLUMN).mean().alias("ctr"),
                pl.col("event_date_bj").min().alias("min_date"),
                pl.col("event_date_bj").max().alias("max_date"),
            )
            .collect()
            .row(0, named=True)
        )

        summary["split"] = split_name
        rows.append(summary)

    result = pl.DataFrame(rows).select(
        "split",
        "rows",
        "clicks",
        "ctr",
        "min_date",
        "max_date",
    )

    result.write_csv(OUTPUT_DIR / "split_summary.csv")

    print()
    print("=" * 76)
    print("SPLIT SUMMARY")
    print("=" * 76)
    print(result)

    return result


def collect_train_vocabularies() -> dict:
    """
    Collect unique non-null training values for every categorical feature.

    These values are used only to measure unseen categories in val/test.
    The permanent encoders will be created in step 07.
    """
    print()
    print("Collecting training categorical vocabularies...")

    vocabulary_frame = (
        load_split("train")
        .select(
            [
                pl.col(feature)
                .drop_nulls()
                .unique()
                .implode()
                .alias(feature)
                for feature in CATEGORICAL_FEATURES
            ]
        )
        .collect()
    )

    return {
        feature: vocabulary_frame[feature][0]
        for feature in CATEGORICAL_FEATURES
    }


def profile_categorical_features() -> pl.DataFrame:
    print()
    print("=" * 76)
    print("CATEGORICAL FEATURE PROFILE")
    print("=" * 76)

    train = load_split("train")

    train_stats = (
        train.select(
            pl.len().alias("rows"),
            *[
                pl.col(feature)
                .drop_nulls()
                .n_unique()
                .alias(f"{feature}__unique")
                for feature in CATEGORICAL_FEATURES
            ],
            *[
                pl.col(feature)
                .null_count()
                .alias(f"{feature}__null")
                for feature in CATEGORICAL_FEATURES
            ],
        )
        .collect()
        .row(0, named=True)
    )

    vocabularies = collect_train_vocabularies()
    split_stats = {}

    for split_name in ("val", "test"):
        data = load_split(split_name)

        expressions = [pl.len().alias("rows")]

        for feature in CATEGORICAL_FEATURES:
            unseen_condition = (
                pl.col(feature).is_not_null()
                & ~pl.col(feature).is_in(
                    vocabularies[feature].implode()
                )
            )

            expressions.extend(
                [
                    pl.col(feature)
                    .null_count()
                    .alias(f"{feature}__null"),
                    unseen_condition
                    .sum()
                    .alias(f"{feature}__unseen_rows"),
                    pl.col(feature)
                    .filter(unseen_condition)
                    .n_unique()
                    .alias(f"{feature}__unseen_unique"),
                ]
            )

        split_stats[split_name] = (
            data.select(expressions)
            .collect()
            .row(0, named=True)
        )

    train_rows = int(train_stats["rows"])
    val_rows = int(split_stats["val"]["rows"])
    test_rows = int(split_stats["test"]["rows"])

    rows = []

    for feature in CATEGORICAL_FEATURES:
        train_null = int(train_stats[f"{feature}__null"])
        val_null = int(split_stats["val"][f"{feature}__null"])
        test_null = int(split_stats["test"][f"{feature}__null"])

        val_unseen = int(
            split_stats["val"][f"{feature}__unseen_rows"]
        )
        test_unseen = int(
            split_stats["test"][f"{feature}__unseen_rows"]
        )

        rows.append(
            {
                "feature": feature,
                "train_unique_non_null": int(
                    train_stats[f"{feature}__unique"]
                ),
                "train_null_rows": train_null,
                "train_null_rate": train_null / train_rows,
                "val_null_rows": val_null,
                "val_unseen_rows": val_unseen,
                "val_unknown_rate": (
                    val_null + val_unseen
                ) / val_rows,
                "val_unseen_unique": int(
                    split_stats["val"][
                        f"{feature}__unseen_unique"
                    ]
                ),
                "test_null_rows": test_null,
                "test_unseen_rows": test_unseen,
                "test_unknown_rate": (
                    test_null + test_unseen
                ) / test_rows,
                "test_unseen_unique": int(
                    split_stats["test"][
                        f"{feature}__unseen_unique"
                    ]
                ),
            }
        )

    result = pl.DataFrame(rows)

    result.write_csv(
        OUTPUT_DIR / "categorical_profile.csv"
    )

    print(result)

    return result


def profile_numeric_features() -> pl.DataFrame:
    print()
    print("=" * 76)
    print("TRAIN NUMERIC FEATURE PROFILE")
    print("=" * 76)

    train = load_split("train")

    expressions = []

    for feature in NUMERIC_FEATURES:
        expressions.extend(
            [
                pl.col(feature)
                .null_count()
                .alias(f"{feature}__null"),
                pl.col(feature)
                .min()
                .cast(pl.Float64)
                .alias(f"{feature}__min"),
                pl.col(feature)
                .mean()
                .alias(f"{feature}__mean"),
                pl.col(feature)
                .median()
                .cast(pl.Float64)
                .alias(f"{feature}__median"),
                pl.col(feature)
                .quantile(0.95)
                .cast(pl.Float64)
                .alias(f"{feature}__p95"),
                pl.col(feature)
                .quantile(0.99)
                .cast(pl.Float64)
                .alias(f"{feature}__p99"),
                pl.col(feature)
                .max()
                .cast(pl.Float64)
                .alias(f"{feature}__max"),
            ]
        )

    stats = (
        train.select(
            pl.len().alias("rows"),
            *expressions,
        )
        .collect()
        .row(0, named=True)
    )

    total_rows = int(stats["rows"])
    rows = []

    for feature in NUMERIC_FEATURES:
        null_rows = int(stats[f"{feature}__null"])

        rows.append(
            {
                "feature": feature,
                "null_rows": null_rows,
                "null_rate": null_rows / total_rows,
                "min": stats[f"{feature}__min"],
                "mean": stats[f"{feature}__mean"],
                "median": stats[f"{feature}__median"],
                "p95": stats[f"{feature}__p95"],
                "p99": stats[f"{feature}__p99"],
                "max": stats[f"{feature}__max"],
            }
        )

    result = pl.DataFrame(rows)

    result.write_csv(
        OUTPUT_DIR / "numeric_profile.csv"
    )

    print(result)

    return result


def write_feature_manifest(schema: dict) -> None:
    manifest = {
        "label": LABEL_COLUMN,
        "categorical_features": CATEGORICAL_FEATURES,
        "binary_features": BINARY_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "metadata_columns": METADATA_COLUMNS,
        "excluded_redundant_columns": (
            EXCLUDED_REDUNDANT_COLUMNS
        ),
        "dtypes": {
            column: str(dtype)
            for column, dtype in schema.items()
        },
        "encoding_plan": {
            "categorical": (
                "Fit vocabulary on train only; reserve 0 for PAD "
                "and 1 for UNKNOWN; known categories start at 2."
            ),
            "binary": (
                "Keep as 0/1 and cast to Int8."
            ),
            "behavior_counts": (
                "Apply log1p before models that require scaled "
                "numeric inputs."
            ),
            "log_price": (
                "Use log_price instead of price_raw or price_clean."
            ),
            "missing_recency": (
                "Fill missing recency with 15 days and retain "
                "history-presence indicators."
            ),
            "validation_and_test": (
                "Apply train-fitted transformations only."
            ),
        },
    }

    output_path = OUTPUT_DIR / "feature_manifest.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nFeature manifest saved to: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    schema = validate_schema()
    build_split_summary()
    profile_categorical_features()
    profile_numeric_features()
    write_feature_manifest(schema)

    print()
    print("=" * 76)
    print("MODEL FEATURE PROFILING COMPLETED")
    print("=" * 76)
    print(f"Reports are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()