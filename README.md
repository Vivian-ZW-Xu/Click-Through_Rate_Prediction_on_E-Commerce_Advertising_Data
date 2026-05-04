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

- **Version A** (26 cols): keeps orphan impressions, fills their missing profile fields with `0`, adds a `has_profile` flag column.
- **Version B** (25 cols): drops orphan impressions entirely; no flag column.
