import polars as pl
from pathlib import Path
import time


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


SCHEMAS = {
    "raw_sample.csv": {
        "user": pl.Int64,
        "time_stamp": pl.Int64,
        "adgroup_id": pl.Int64,
        "pid": pl.Utf8,
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
        "new_user_class_level": pl.Float64,
    },
    "behavior_log.csv": {
        "user": pl.Int64,
        "time_stamp": pl.Int64,
        "btag": pl.Utf8,
        "cate": pl.Int64,
        "brand": pl.Int64,
    },
}


def convertSmall(csvPath, parquetPath, schema):
    print(f"[small] {csvPath.name}")
    start = time.time()
    df = pl.read_csv(csvPath, schema_overrides=schema, ignore_errors=True)
    df.write_parquet(parquetPath, compression="snappy")
    print(f"  rows={df.height:,}  time={time.time()-start:.1f}s")


def convertLarge(csvPath, parquetPath, schema):
    print(f"[large/streaming] {csvPath.name}")
    start = time.time()
    (
        pl.scan_csv(csvPath, schema_overrides=schema, ignore_errors=True)
        .sink_parquet(parquetPath, compression="snappy")
    )
    nRows = pl.scan_parquet(parquetPath).select(pl.len()).collect().item()
    print(f"  rows={nRows:,}  time={time.time()-start:.1f}s")


def main():
    smallFiles = ["ad_feature.csv", "user_profile.csv", "raw_sample.csv"]
    largeFiles = ["behavior_log.csv"]

    for name in smallFiles:
        csvPath = RAW_DIR / name
        parquetPath = OUT_DIR / name.replace(".csv", ".parquet")
        if parquetPath.exists():
            print(f"skip {name} (already converted)")
            continue
        convertSmall(csvPath, parquetPath, SCHEMAS[name])

    for name in largeFiles:
        csvPath = RAW_DIR / name
        parquetPath = OUT_DIR / name.replace(".csv", ".parquet")
        if parquetPath.exists():
            print(f"skip {name} (already converted)")
            continue
        convertLarge(csvPath, parquetPath, SCHEMAS[name])

    print("\nDone. Files in data/processed/:")
    for p in sorted(OUT_DIR.glob("*.parquet")):
        sizeMb = p.stat().st_size / (1024**2)
        print(f"  {p.name}  {sizeMb:.1f} MB")


if __name__ == "__main__":
    main()
