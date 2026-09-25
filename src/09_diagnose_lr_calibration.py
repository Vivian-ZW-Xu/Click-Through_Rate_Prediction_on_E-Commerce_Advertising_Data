from pathlib import Path

import numpy as np
import polars as pl


PRED_DIR = Path(
    "artifacts/predictions/logistic_regression"
)
FEATURE_DIR = Path(
    "data/processed/features"
)
OUTPUT_DIR = Path(
    "results/logistic_regression/calibration_diagnostics"
)


def summarize(
    data: pl.DataFrame,
    groups: list[str],
) -> pl.DataFrame:
    """Calculate probability quality for traffic groups."""
    return (
        data.with_columns(
            pl.col("prediction")
            .clip(1e-7, 1 - 1e-7)
            .alias("p")
        )
        .group_by(
            groups,
            maintain_order=True,
        )
        .agg(
            pl.len().alias("rows"),
            pl.col("clk")
            .mean()
            .alias("actual_ctr"),
            pl.col("prediction")
            .mean()
            .alias("mean_prediction"),
            (
                -(
                    pl.col("clk")
                    * pl.col("p").log()
                    + (1 - pl.col("clk"))
                    * (1 - pl.col("p")).log()
                ).mean()
            ).alias("logloss"),
        )
        .with_columns(
            (
                pl.col("actual_ctr")
                / pl.col("mean_prediction")
            ).alias("copc"),
            (
                100
                * (
                    pl.col("mean_prediction")
                    / pl.col("actual_ctr")
                    - 1
                )
            ).alias("prediction_bias_pct"),
        )
        .sort(groups)
    )


def load_split(
    split: str,
) -> pl.DataFrame:
    predictions = pl.read_parquet(
        PRED_DIR
        / f"{split}_predictions.parquet"
    )

    context = (
        pl.scan_parquet(
            FEATURE_DIR / f"{split}.parquet"
        )
        .select(
            "impression_id",
            "hour",
            "pid_raw",
        )
        .collect()
    )

    data = predictions.join(
        context,
        on="impression_id",
        how="inner",
        validate="1:1",
    )

    if data.height != predictions.height:
        raise ValueError(
            f"{split}: context join changed "
            "the row count"
        )

    # Equal-frequency prediction deciles:
    # 1 is lowest, 10 is highest.
    order = np.argsort(
        data["prediction"].to_numpy(),
        kind="stable",
    )

    decile = np.empty(
        data.height,
        dtype=np.int8,
    )

    decile[order] = (
        np.arange(data.height)
        * 10
        // data.height
        + 1
    )

    return data.with_columns(
        pl.lit(split).alias("split"),
        pl.Series(
            "prediction_decile",
            decile,
        ),
    )


def main() -> None:
    print("=" * 74)
    print("LR CALIBRATION DIAGNOSTICS")
    print("=" * 74)
    print(
        "COPC = actual CTR / mean prediction; "
        "the ideal value is 1.0."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_tables = []
    hourly_tables = []
    pid_tables = []
    decile_tables = []

    for split in ("val", "test"):
        print(f"Loading {split}...")

        data = load_split(split)

        overall_tables.append(
            summarize(
                data,
                ["split"],
            )
        )

        hourly_tables.append(
            summarize(
                data,
                ["split", "hour"],
            )
        )

        pid_tables.append(
            summarize(
                data,
                ["split", "pid_raw"],
            )
        )

        decile_tables.append(
            summarize(
                data,
                [
                    "split",
                    "prediction_decile",
                ],
            )
        )

    overall = pl.concat(
        overall_tables
    )

    hourly = pl.concat(
        hourly_tables
    )

    by_pid = pl.concat(
        pid_tables
    )

    deciles = pl.concat(
        decile_tables
    )

    overall.write_csv(
        OUTPUT_DIR / "overall.csv"
    )

    hourly.write_csv(
        OUTPUT_DIR / "by_hour.csv"
    )

    by_pid.write_csv(
        OUTPUT_DIR / "by_pid.csv"
    )

    deciles.write_csv(
        OUTPUT_DIR / "by_decile.csv"
    )

    with pl.Config(
        tbl_rows=25,
        tbl_cols=10,
    ):
        print("\nOverall:")
        print(overall)

        print(
            "\nBy prediction decile:"
        )
        print(deciles)

        print("\nBy pid:")
        print(by_pid)

    print(
        f"\nSaved diagnostic tables to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()