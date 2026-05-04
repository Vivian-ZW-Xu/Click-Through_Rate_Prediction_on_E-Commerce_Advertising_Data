# Alimama CTR Project

Dependencies in `requirements.txt`. Python 3.11.

## Loading the Preprocessed Data

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
