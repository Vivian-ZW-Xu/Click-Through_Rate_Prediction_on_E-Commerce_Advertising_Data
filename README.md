# Click-Through Rate Prediction on E-Commerce Advertising Data

An empirical study comparing five CTR prediction models — Logistic Regression, LightGBM,
Wide & Deep, DeepFM, and DIN — on the Alibaba Taobao Display Advertising Dataset.

**Final paper for DS-GA 1003 (Spring 2026), New York University.**

- **Authors**: Zhuwei Xu, Chuhan Ku (Center for Data Science, NYU)
- **Paper**: [`MLProject.pdf`](MLProject.pdf)


## Headline Results

Test-set performance on Alibaba Taobao Display Advertising (2.85M test impressions, 5.04% CTR):

| Model        | AUC    | LogLoss | GAUC   |
|--------------|--------|---------|--------|
| LR           | 0.5779 | 0.2552  | 0.5382 |
| LightGBM     | 0.5973 | 0.1985  | 0.5439 |
| Wide & Deep  | 0.6123 | 0.2067  | 0.5451 |
| DeepFM       | 0.6211 | 0.2014  | 0.5504 |
| **DIN**      | **0.6235** | **0.1957** | **0.5525** |

Key findings (see paper):

- **60% of clicks are missed by every model**, concentrating on cold-start users
- **DIN's attention favors semantic relevance over recency** (flat distribution across positions)
- **Sample-complexity curves** show that classical baselines saturate well below deep
  models even at full training data — the remaining headroom is *representational*,
  not data-driven
- Classical (LR, LightGBM) and deep (DeepFM, DIN) models capture qualitatively
  different click sub-populations, suggesting ensembling has untapped potential


## Repository Structure

```
final_project/
├── data/                       (gitignored — see Reproducing below)
│   ├── raw/                    raw CSVs from Tianchi
│   └── processed/              parquet conversions and wide tables
├── notebooks/                  preprocessing pipeline
│   ├── preprocessing.ipynb     joins, cleaning, encoding, time split
│   ├── behavior_features.ipynb user_total_* + user_cate_ctr aggregates
│   └── din_sequences.ipynb     50-step behavior sequences for DIN
├── scripts/                    training + analysis
│   ├── train_lr.py             classical baseline (sklearn SGDClassifier)
│   ├── train_lgbm.py           LightGBM with native categoricals
│   ├── train_wide_deep.py      Wide & Deep
│   ├── train_deepfm.py         DeepFM (FM + DNN)
│   ├── train_din.py            DIN (attention over behavior sequence)
│   └── analysis.py             D1–D4 + sample complexity + per-segment AUC
├── checkpoints/                (gitignored — model weights ~1 GB)
├── figures/
│   ├── dataset/                user_activity, ctr_by_hour
│   └── analysis/               D1a–d, D2a–c, D3, D4a–c, segment, sample_complexity
├── convert.py                  CSV → parquet
├── requirements.txt
├── MLProject.pdf               Final paper (8 pages + appendix)
└── README.md
```


## Models

We compare 5 CTR models spanning four sources of representational capacity:

| Model        | Adds                                | Why included                                                             |
|--------------|-------------------------------------|--------------------------------------------------------------------------|
| LR           | linear weights                      | Interpretable lower-bound baseline                                       |
| LightGBM     | + nonlinearity (tree splits)        | Isolates value of nonlinearity without embeddings                        |
| Wide & Deep  | + dense ID embeddings               | Tests gain from learned embeddings on top of nonlinearity                |
| DeepFM       | + automatic feature interaction (FM) | Tests if learned FM beats hand-crafted Wide & Deep crosses              |
| DIN          | + attention over behavior sequence  | Tests gain from sequence-aware user-interest modeling                    |

CTR aggregate features (`user_ctr`, `ad_ctr`, etc.) are used for LR/LightGBM but
**excluded** from the three deep models, because combining high-cardinality ID
embeddings with per-entity CTR statistics derived from the same training labels
causes severe within-train label leakage (train AUC > 0.9, val AUC < 0.6).


## Splits

Strict time-based, no shuffling:

- **Train**: 2017-05-05 12:00 — 2017-05-12 00:00 UTC (20.4M impressions, 77%)
- **Val**:   2017-05-12 (3.27M, 12%)
- **Test**:  2017-05-13 00:00 — 2017-05-13 11:59 UTC (2.85M, 11%)

All aggregate statistics are computed on the training split only and joined to
val/test as features, ensuring no test-time labels enter the feature pipeline.


## Reproducing the Pipeline

The processed parquets (~5 GB total) and trained checkpoints (~1 GB) are not
committed to the repo (gitignored). Reproduction takes roughly an hour end-to-end
on a workstation with an Apple M4 Max or comparable GPU.

### 1. Download raw data

[Alibaba Tianchi dataset](https://tianchi.aliyun.com/dataset/56) → place under
`data/raw/`:

- `raw_sample.csv` (impression logs)
- `ad_feature.csv` (ad metadata)
- `user_profile.csv` (user demographics)
- `behavior_log.csv` (~22 GB, behavioral history)

### 2. Convert CSVs to parquet

```bash
python convert.py
```

Writes `data/processed/{raw_sample,ad_feature,user_profile,behavior_log}.parquet`.

### 3. Run preprocessing notebooks (in order)

Each notebook overwrites `data/processed/wide/*.parquet` with additional columns:

```
notebooks/preprocessing.ipynb       → joins, cleaning, encoding, basic CTR aggregates
                                      (output: A=26 cols, B=25 cols)
notebooks/behavior_features.ipynb   → adds user_total_* and user_cate_ctr
                                      (output: A=34, B=33)
notebooks/din_sequences.ipynb       → adds behavior_seq (50-step cate history)
                                      (output: A=35, B=34)
```

The paper uses **Version A** (with `has_profile` flag for orphan users); Version B
is provided for ablation.

### 4. Train models

```bash
python scripts/train_lr.py            # ~5 min, CPU
python scripts/train_lgbm.py          # ~10 min, CPU
python scripts/train_wide_deep.py     # ~20 min, MPS/GPU
python scripts/train_deepfm.py        # ~20 min, MPS/GPU
python scripts/train_din.py           # ~30 min, MPS/GPU
```

Each script saves the best-val-AUC checkpoint to `checkpoints/*.pt` (or
`checkpoints/*.txt`/`*.pkl`) plus per-model test predictions to
`checkpoints/preds_*.npy`.

### 5. Run analysis

```bash
python scripts/analysis.py
```

Produces all figures under `figures/analysis/` (D1–D4, sample complexity,
per-segment AUC) and prints summary numbers to stdout.


## Dependencies

See `requirements.txt`. Tested with Python 3.11.

Key libraries:
- `polars` — fast parquet I/O and aggregation
- `torch` + `deepctr-torch` — DIN, DeepFM, Wide & Deep
- `lightgbm` — LightGBM (requires `libomp` on macOS: `brew install libomp`)
- `scikit-learn` — LR, metrics
- `shap` — SHAP analysis for D4b
- `matplotlib` — figures

Deep models trained with Apple MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1` set in
training scripts to allow CPU fallback for unsupported ops).


## Authors

- **Zhuwei Xu** — `zx2188@nyu.edu`, Center for Data Science, NYU
- **Chuhan Ku** — `ck3504@nyu.edu`, Center for Data Science, NYU

DS-GA 1003 *Machine Learning* (Spring 2026), New York University.
