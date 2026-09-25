import json
from pathlib import Path

import polars as pl


INPUT_DIR = Path("data/processed_v2/features")
OUTPUT_DIR = Path("data/processed_v2/model_input")
ENCODER_DIR = Path("artifacts_v2/encoders")
MAPPING_DIR = ENCODER_DIR / "mappings"
REPORT_DIR = Path("results_v2/model_input")

SPLITS = ("train", "val", "test")

EXPECTED_ROWS = {
    "train": 20_015_245,
    "val": 3_234_051,
    "test": 3_308_665,
}

PAD_INDEX = 0
UNKNOWN_INDEX = 1
FIRST_KNOWN_INDEX = 2
MISSING_RECENCY_VALUE = 15.0

CATEGORICAL_FEATURES = {
    "user_raw": "user_idx",
    "adgroup_id_raw": "adgroup_idx",
    "pid_raw": "pid_idx",
    "cate_id_raw": "cate_idx",
    "campaign_id_raw": "campaign_idx",
    "customer_raw": "customer_idx",
    "brand_raw": "brand_idx",
    "cms_segid": "cms_segid_idx",
    "cms_group_id": "cms_group_idx",
    "final_gender_code": "gender_idx",
    "age_level": "age_idx",
    "pvalue_level": "pvalue_idx",
    "shopping_level": "shopping_idx",
    "occupation": "occupation_idx",
    "new_user_class_level": "new_user_class_idx",
    "hour": "hour_idx",
    "day_of_week": "day_of_week_idx",
}

FIXED_DOMAINS = {
    "hour": list(range(24)),
    # Polars dt.weekday() uses ISO weekday numbering: Monday=1, Sunday=7.
    "day_of_week": list(range(1, 8)),
}

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
        "pv_count",
        "cart_count",
        "fav_count",
        "buy_count",
    )
]

LOG_COUNT_FEATURES = {
    feature: f"log1p_{feature}"
    for feature in BEHAVIOR_COUNT_FEATURES
}

RECENCY_FEATURES = [
    "days_since_last_event_14d",
    "days_since_last_buy_14d",
]

METADATA_COLUMNS = [
    "impression_id",
    "event_date_bj",
]

LABEL_COLUMN = "clk"


def input_path(split_name: str) -> Path:
    return INPUT_DIR / f"{split_name}.parquet"


def output_path(split_name: str) -> Path:
    return OUTPUT_DIR / f"{split_name}.parquet"


def mapping_path(feature: str) -> Path:
    return MAPPING_DIR / f"{feature}_mapping.parquet"


def validate_inputs() -> dict:
    for split_name in SPLITS:
        path = input_path(split_name)
        if not path.exists():
            raise FileNotFoundError(f"Missing feature table: {path}")

    schemas = {
        split_name: pl.scan_parquet(input_path(split_name)).collect_schema()
        for split_name in SPLITS
    }

    if schemas["val"] != schemas["train"]:
        raise ValueError("Validation schema does not match training schema")

    if schemas["test"] != schemas["train"]:
        raise ValueError("Test schema does not match training schema")

    required = (
        set(CATEGORICAL_FEATURES)
        | set(BINARY_FEATURES)
        | set(BEHAVIOR_COUNT_FEATURES)
        | set(RECENCY_FEATURES)
        | set(METADATA_COLUMNS)
        | {"log_price", LABEL_COLUMN}
    )

    missing = sorted(required - set(schemas["train"].names()))
    if missing:
        raise ValueError(f"Required input columns are missing: {missing}")

    return schemas["train"]


def collect_train_vocabularies() -> dict:
    train_fit_features = [
        feature
        for feature in CATEGORICAL_FEATURES
        if feature not in FIXED_DOMAINS
    ]

    print("Collecting train-only categorical vocabularies...")

    vocab_frame = (
        pl.scan_parquet(input_path("train"))
        .select(
            [
                pl.col(feature)
                .drop_nulls()
                .unique()
                .implode()
                .alias(feature)
                for feature in train_fit_features
            ]
        )
        .collect()
    )

    return {
        feature: vocab_frame[feature][0]
        for feature in train_fit_features
    }


def build_mappings(train_schema: dict) -> dict:
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)

    vocabularies = collect_train_vocabularies()
    vocabulary_sizes = {}

    print("\nWriting categorical mapping tables...")

    for raw_feature, encoded_feature in CATEGORICAL_FEATURES.items():
        if raw_feature in FIXED_DOMAINS:
            values = pl.Series(
                raw_feature,
                FIXED_DOMAINS[raw_feature],
                dtype=train_schema[raw_feature],
            )
        else:
            values = vocabularies[raw_feature]

        mapping = (
            pl.DataFrame({raw_feature: values})
            .sort(raw_feature)
            .with_row_index(
                encoded_feature,
                offset=FIRST_KNOWN_INDEX,
            )
            .select(raw_feature, encoded_feature)
        )

        if mapping[raw_feature].null_count() != 0:
            raise ValueError(
                f"Null value found in mapping for {raw_feature}"
            )

        if mapping[raw_feature].n_unique() != mapping.height:
            raise ValueError(
                f"Duplicate value found in mapping for {raw_feature}"
            )

        mapping.write_parquet(
            mapping_path(raw_feature),
            compression="zstd",
            statistics=True,
        )

        vocabulary_sizes[encoded_feature] = mapping.height + 2

        print(
            f"  {raw_feature:<24} -> {encoded_feature:<24} "
            f"known={mapping.height:,}, "
            f"vocabulary_size={mapping.height + 2:,}"
        )

    return vocabulary_sizes


def get_train_log_price_median() -> float:
    median = (
        pl.scan_parquet(input_path("train"))
        .select(pl.col("log_price").median())
        .collect()
        .item()
    )

    if median is None:
        raise ValueError("Training log_price median is null")

    return float(median)


def encode_split(
    split_name: str,
    log_price_median: float,
) -> None:
    print(f"\nEncoding {split_name}...")

    data = pl.scan_parquet(input_path(split_name))

    for raw_feature, encoded_feature in CATEGORICAL_FEATURES.items():
        mapping = pl.scan_parquet(mapping_path(raw_feature))

        data = (
            data.join(
                mapping,
                on=raw_feature,
                how="left",
            )
            .with_columns(
                pl.col(encoded_feature)
                .fill_null(UNKNOWN_INDEX)
                .cast(pl.UInt32)
            )
        )

    output_expressions = [
        pl.col("impression_id"),
        pl.col("event_date_bj"),
        pl.col(LABEL_COLUMN).cast(pl.UInt8),
    ]

    output_expressions.extend(
        [
            pl.col(encoded_feature).cast(pl.UInt32)
            for encoded_feature in CATEGORICAL_FEATURES.values()
        ]
    )

    output_expressions.extend(
        [
            pl.col(feature).cast(pl.UInt8)
            for feature in BINARY_FEATURES
        ]
    )

    output_expressions.append(
        pl.col("log_price")
        .fill_null(log_price_median)
        .cast(pl.Float32)
        .alias("log_price")
    )

    output_expressions.extend(
        [
            pl.col(raw_feature)
            .cast(pl.Float64)
            .log1p()
            .cast(pl.Float32)
            .alias(output_feature)
            for raw_feature, output_feature in LOG_COUNT_FEATURES.items()
        ]
    )

    output_expressions.extend(
        [
            pl.col(feature)
            .fill_null(MISSING_RECENCY_VALUE)
            .cast(pl.Float32)
            .alias(feature)
            for feature in RECENCY_FEATURES
        ]
    )

    encoded = data.select(output_expressions)

    encoded.sink_parquet(
        output_path(split_name),
        compression="zstd",
        statistics=True,
    )

    size_mb = output_path(split_name).stat().st_size / (1024 * 1024)
    print(
        f"Saved {output_path(split_name)}: {size_mb:,.1f} MB"
    )


def validate_output_split(
    split_name: str,
    vocabulary_sizes: dict,
) -> dict:
    data = pl.scan_parquet(output_path(split_name))

    encoded_features = list(CATEGORICAL_FEATURES.values())
    numeric_features = [
        "log_price",
        *LOG_COUNT_FEATURES.values(),
        *RECENCY_FEATURES,
    ]
    model_features = [
        *encoded_features,
        *BINARY_FEATURES,
        *numeric_features,
    ]

    summary = (
        data.select(
            pl.len().alias("rows"),
            pl.col("impression_id")
            .n_unique()
            .alias("unique_impressions"),
            pl.col(LABEL_COLUMN).mean().alias("ctr"),
            (pl.col("user_idx") == UNKNOWN_INDEX)
            .mean()
            .alias("user_unknown_rate"),
            (pl.col("adgroup_idx") == UNKNOWN_INDEX)
            .mean()
            .alias("ad_unknown_rate"),
            (pl.col("brand_idx") == UNKNOWN_INDEX)
            .mean()
            .alias("brand_unknown_rate"),
            (pl.col("day_of_week_idx") == UNKNOWN_INDEX)
            .mean()
            .alias("day_unknown_rate"),
        )
        .collect()
        .row(0, named=True)
    )

    if int(summary["rows"]) != EXPECTED_ROWS[split_name]:
        raise ValueError(
            f"{split_name}: unexpected row count "
            f"{summary['rows']}"
        )

    if int(summary["unique_impressions"]) != EXPECTED_ROWS[split_name]:
        raise ValueError(
            f"{split_name}: duplicate impression IDs detected"
        )

    null_counts = (
        data.select(
            [
                pl.col(feature).null_count().alias(feature)
                for feature in model_features
            ]
        )
        .collect()
    )

    total_nulls = sum(
        int(null_counts[feature][0])
        for feature in model_features
    )

    if total_nulls != 0:
        raise ValueError(
            f"{split_name}: model features contain "
            f"{total_nulls:,} null values"
        )

    encoded_ranges = (
        data.select(
            [
                pl.col(feature).min().alias(f"{feature}__min")
                for feature in encoded_features
            ]
            + [
                pl.col(feature).max().alias(f"{feature}__max")
                for feature in encoded_features
            ]
        )
        .collect()
        .row(0, named=True)
    )

    for feature in encoded_features:
        minimum = int(encoded_ranges[f"{feature}__min"])
        maximum = int(encoded_ranges[f"{feature}__max"])

        if minimum < UNKNOWN_INDEX:
            raise ValueError(
                f"{split_name}: {feature} contains PAD or "
                f"negative index {minimum}"
            )

        if maximum >= vocabulary_sizes[feature]:
            raise ValueError(
                f"{split_name}: {feature} index {maximum} "
                f"exceeds vocabulary size "
                f"{vocabulary_sizes[feature]}"
            )

    if summary["day_unknown_rate"] != 0.0:
        raise ValueError(
            f"{split_name}: calendar weekday was mapped to UNKNOWN"
        )

    return {
        "split": split_name,
        **summary,
    }


def validate_outputs(vocabulary_sizes: dict) -> pl.DataFrame:
    print("\n" + "=" * 76)
    print("MODEL INPUT VALIDATION")
    print("=" * 76)

    schemas = {
        split_name: pl.scan_parquet(
            output_path(split_name)
        ).collect_schema()
        for split_name in SPLITS
    }

    if schemas["val"] != schemas["train"]:
        raise ValueError("Encoded validation schema differs from train")

    if schemas["test"] != schemas["train"]:
        raise ValueError("Encoded test schema differs from train")

    rows = [
        validate_output_split(split_name, vocabulary_sizes)
        for split_name in SPLITS
    ]

    result = pl.DataFrame(rows).select(
        "split",
        "rows",
        "unique_impressions",
        "ctr",
        "user_unknown_rate",
        "ad_unknown_rate",
        "brand_unknown_rate",
        "day_unknown_rate",
    )

    result.write_csv(REPORT_DIR / "model_input_summary.csv")
    print(result)
    print("\nAll model-input validation checks passed.")

    return result


def write_manifest(
    train_schema: dict,
    vocabulary_sizes: dict,
    log_price_median: float,
) -> None:
    manifest = {
        "reserved_indices": {
            "PAD": PAD_INDEX,
            "UNKNOWN": UNKNOWN_INDEX,
            "first_known_index": FIRST_KNOWN_INDEX,
        },
        "categorical_source_to_encoded": CATEGORICAL_FEATURES,
        "categorical_features": list(CATEGORICAL_FEATURES.values()),
        "vocabulary_sizes": vocabulary_sizes,
        "fixed_domains": FIXED_DOMAINS,
        "binary_features": BINARY_FEATURES,
        "numeric_features": [
            "log_price",
            *LOG_COUNT_FEATURES.values(),
            *RECENCY_FEATURES,
        ],
        "behavior_count_source_to_log1p": LOG_COUNT_FEATURES,
        "metadata_columns": METADATA_COLUMNS,
        "label": LABEL_COLUMN,
        "fill_values": {
            "log_price": log_price_median,
            "days_since_last_event_14d": MISSING_RECENCY_VALUE,
            "days_since_last_buy_14d": MISSING_RECENCY_VALUE,
        },
        "training_rules": {
            "categorical_vocabulary": "fit_on_train_only",
            "validation_test_unknown": UNKNOWN_INDEX,
            "calendar_domains": "predefined",
            "numeric_scaling": "fit_inside_each_model_on_train_only",
        },
        "source_dtypes": {
            column: str(dtype)
            for column, dtype in train_schema.items()
        },
    }

    path = ENCODER_DIR / "encoding_manifest.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"Encoding manifest saved to: {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("BUILDING TRAIN-ONLY MODEL ENCODINGS")
    print("=" * 76)

    train_schema = validate_inputs()
    vocabulary_sizes = build_mappings(train_schema)
    log_price_median = get_train_log_price_median()

    print(f"\nTraining log_price median: {log_price_median:.6f}")

    for split_name in SPLITS:
        encode_split(split_name, log_price_median)

    validate_outputs(vocabulary_sizes)
    write_manifest(
        train_schema,
        vocabulary_sizes,
        log_price_median,
    )

    print("\n" + "=" * 76)
    print("COMMON TABULAR PREPROCESSING COMPLETED")
    print("=" * 76)
    print(f"Model inputs: {OUTPUT_DIR}")
    print(f"Encoders:     {ENCODER_DIR}")
    print(f"Reports:      {REPORT_DIR}")


if __name__ == "__main__":
    main()
