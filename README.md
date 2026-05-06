# Alimama CTR Project

Dependencies in `requirements.txt`. Python 3.11.

## Loading the Preprocessed Data

The processed parquet files are not in the repo (~5 GB total). See [Reproducing the Data](#reproducing-the-data) below to regenerate them. Once they exist under `data/processed/wide/`:

```python
import polars as pl
from pathlib import Path

WIDE = Path('../data/processed/wide')

trainA = pl.read_parquet(WIDE / 'trainA.parquet')
valA   = pl.read_parquet(WIDE / 'valA.parquet')
testA  = pl.read_parquet(WIDE / 'testA.parquet')
trainB = pl.read_parquet(WIDE / 'trainB.parquet')
valB   = pl.read_parquet(WIDE / 'valB.parquet')
testB  = pl.read_parquet(WIDE / 'testB.parquet')
```

## A vs B

About 5.8% of impressions in `raw_sample` come from users absent in `user_profile` (orphan users). The two versions handle them differently for an A/B ablation:

- **Version A** (35 cols): keeps orphan impressions, fills their missing profile fields with `0`, adds a `has_profile` flag column.
- **Version B** (34 cols): drops orphan impressions entirely; no flag column.

## Columns

Both versions contain (besides the `clk` label):
- Identifiers and ad metadata: `user`, `adgroup_id`, `pid`, `cate_id`, `campaign_id`, `customer`, `price`, `log_price`
- User profile (8 cols): `cms_segid`, `cms_group_id`, `final_gender_code`, `age_level`, `pvalue_level`, `shopping_level`, `occupation`, `new_user_class_level`
- Time: `time_stamp`, `hour` (Beijing time), `day_of_week`
- Impression aggregates: `user_imp_count`, `user_ctr`, `ad_imp_count`, `ad_ctr`, `cate_ctr`, `user_cate_ctr` (Bayesian smoothed)
- Behavior-log aggregates: `user_total_events`, `user_total_pv`, `user_total_buy`, `user_total_cart`, `user_total_fav`, `user_n_unique_cates`, `user_buy_rate` (smoothed)
- DIN sequence input: `behavior_seq` — list of 50 `int16` cate codes per impression (last position = most recent; left-padded with 0)

Version A additionally has `has_profile` (binary flag).

## Splits

Time-based, UTC midnight cutoffs (no shuffling):

- Train: 2017-05-05 12:00 — 2017-05-12 00:00 UTC (~6.5 days, 77%)
- Val:   2017-05-12 (1 day, 12%)
- Test:  2017-05-13 00:00 — 2017-05-13 11:59 UTC (12 hours, 11%)

See `preprocessing_section.docx` for full preprocessing details.

## Reproducing the Data

The processed parquets (~5 GB total) are not committed to the repo. To regenerate them from scratch:

1. Download the raw CSVs from the [Alibaba Tianchi dataset](https://tianchi.aliyun.com/dataset/56) and place them under `data/raw/`:
   `raw_sample.csv`, `ad_feature.csv`, `user_profile.csv`, `behavior_log.csv`

2. Convert CSVs to parquet:

   ```
   python convert.py
   ```

   This writes `data/processed/{raw_sample,ad_feature,user_profile,behavior_log}.parquet`.

3. Run the notebooks **in order** — each one overwrites `data/processed/wide/*.parquet`:

   1. `notebooks/preprocessing.ipynb` — joins, cleaning, ID encoding, time-based split, basic CTR aggregates (→ A=26 cols, B=25 cols)
   2. `notebooks/behavior_features.ipynb` — adds 7 `user_total_*` features and `user_cate_ctr` (→ A=34, B=33)
   3. `notebooks/din_sequences.ipynb` — adds `behavior_seq` (→ A=35, B=34)

4. Train:

   ```
   python scripts/train_deepfm.py
   python scripts/train_din.py
   ```
