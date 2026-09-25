from datetime import date
from math import isclose
from pathlib import Path

import polars as pl


SCRIPT_VERSION = "identity-check-v2"

LOOKUP_DIR = Path("data/processed/behavior/window_features")
FEATURE_DIR = Path("data/processed/features")

SAMPLE_USERS_PER_DATE = 20

WINDOWS = (1, 3, 7, 14)
METRICS = (
    "event_count",
    "pv_count",
    "cart_count",
    "fav_count",
    "buy_count",
)

COUNT_COLUMNS = [
    f"{metric}_{window}d"
    for window in WINDOWS
    for metric in METRICS
]

FEATURE_COLUMNS = COUNT_COLUMNS + [
    "has_behavior_history_14d",
    "has_buy_history_14d",
    "days_since_last_event_14d",
    "days_since_last_buy_14d",
]

KEY_COLUMNS = [
    "user_raw",
    "event_date_bj",
]


def split_for_date(target_date: date) -> str:
    if target_date <= date(2017, 5, 11):
        return "train"

    if target_date == date(2017, 5, 12):
        return "val"

    if target_date == date(2017, 5, 13):
        return "test"

    raise ValueError(f"Unexpected target date: {target_date}")


def sample_keys_from_impressions(
    target_date: date,
    split_name: str,
) -> pl.DataFrame:
    """
    Sample users that really have an impression on the target date.

    We sample from the final feature table first, so every selected
    user/date pair is guaranteed to exist in the impression data.
    """
    feature_path = FEATURE_DIR / f"{split_name}.parquet"

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing feature table: {feature_path}"
        )

    keys = (
        pl.scan_parquet(feature_path)
        .filter(
            (pl.col("event_date_bj") == target_date)
            & (pl.col("has_behavior_history_14d") == 1)
        )
        .select(KEY_COLUMNS)
        .head(50_000)
        .collect()
        .unique(
            subset=KEY_COLUMNS,
            maintain_order=True,
        )
        .head(SAMPLE_USERS_PER_DATE)
    )

    if keys.height != SAMPLE_USERS_PER_DATE:
        raise ValueError(
            f"{target_date}: expected "
            f"{SAMPLE_USERS_PER_DATE} sampled users, "
            f"but found only {keys.height}"
        )

    return keys


def load_expected_rows() -> pl.DataFrame:
    """
    Read the correct behavior features directly from the lookup tables.
    """
    frames = []

    for day in range(6, 14):
        target_date = date(2017, 5, day)
        split_name = split_for_date(target_date)

        lookup_path = (
            LOOKUP_DIR
            / f"user_behavior_{target_date}.parquet"
        )

        if not lookup_path.exists():
            raise FileNotFoundError(
                f"Missing behavior lookup: {lookup_path}"
            )

        keys = sample_keys_from_impressions(
            target_date,
            split_name,
        )

        expected = (
            pl.scan_parquet(lookup_path)
            .join(
                keys.lazy(),
                on=KEY_COLUMNS,
                how="inner",
            )
            .select(
                KEY_COLUMNS + FEATURE_COLUMNS
            )
            .collect()
            .with_columns(
                pl.lit(split_name).alias("split")
            )
        )

        if expected.height != SAMPLE_USERS_PER_DATE:
            raise ValueError(
                f"{target_date}: sampled "
                f"{SAMPLE_USERS_PER_DATE} impression users, "
                f"but only {expected.height} were found "
                "in the behavior lookup"
            )

        frames.append(expected)

    return pl.concat(
        frames,
        how="vertical",
    )


def load_observed_rows(
    expected: pl.DataFrame,
) -> pl.DataFrame:
    """
    Read behavior features stored in the final train/val/test tables.
    """
    frames = []

    for split_name in ("train", "val", "test"):
        feature_path = (
            FEATURE_DIR
            / f"{split_name}.parquet"
        )

        keys = (
            expected
            .filter(pl.col("split") == split_name)
            .select(KEY_COLUMNS)
        )

        observed = (
            pl.scan_parquet(feature_path)
            .join(
                keys.lazy(),
                on=KEY_COLUMNS,
                how="inner",
            )
            .select(
                KEY_COLUMNS + FEATURE_COLUMNS
            )
            .unique()
            .collect()
        )

        frames.append(observed)

    return pl.concat(
        frames,
        how="vertical",
    )


def values_match(expected, observed) -> bool:
    if expected is None or observed is None:
        return (
            expected is None
            and observed is None
        )

    if (
        isinstance(expected, float)
        or isinstance(observed, float)
    ):
        return isclose(
            float(expected),
            float(observed),
            abs_tol=1e-5,
        )

    return expected == observed


def validate_identity(
    expected: pl.DataFrame,
    observed: pl.DataFrame,
) -> None:
    observed_by_key = {
        (
            row["user_raw"],
            row["event_date_bj"],
        ): row
        for row in observed.iter_rows(named=True)
    }

    errors = []

    for expected_row in expected.iter_rows(named=True):
        key = (
            expected_row["user_raw"],
            expected_row["event_date_bj"],
        )

        observed_row = observed_by_key.get(key)

        if observed_row is None:
            errors.append(
                f"Missing final feature row for {key}"
            )
            continue

        for column in FEATURE_COLUMNS:
            if not values_match(
                expected_row[column],
                observed_row[column],
            ):
                errors.append(
                    f"key={key}, "
                    f"column={column}, "
                    f"expected={expected_row[column]}, "
                    f"observed={observed_row[column]}"
                )

    if errors:
        preview = "\n".join(errors[:20])

        raise AssertionError(
            f"Identity check failed with "
            f"{len(errors)} mismatches.\n"
            f"First mismatches:\n{preview}"
        )


def main() -> None:
    print("=" * 70)
    print("BEHAVIOR IDENTITY CHECK")
    print("=" * 70)
    print(f"Script version: {SCRIPT_VERSION}")
    print(
        "Sampling users from real impressions, then checking "
        "their stored behavior against the same-user lookup."
    )

    expected = load_expected_rows()
    observed = load_observed_rows(expected)

    validate_identity(
        expected,
        observed,
    )

    print()
    print(
        f"Checked {expected.height:,} "
        "user/date pairs."
    )
    print(
        "All sampled behavior features belong "
        "to the correct raw user ID."
    )


if __name__ == "__main__":
    main()