# Point-in-Time CTR Prediction on E-Commerce Advertising Data

A leakage-free CTR prediction and evaluation system built on the Alibaba Taobao Display Advertising Dataset.

The project separates information gain, model-family gain, sequence gain, calibration, cold-start robustness, temporal robustness, and engineering cost.

## Data Protocol

All date boundaries use Beijing time (`Asia/Shanghai`).

| Split | Beijing dates | Purpose |
|---|---|---|
| Train | 2017-05-06 to 2017-05-11 | model fitting |
| Validation | 2017-05-12 | model selection and diagnosis |
| Test | 2017-05-13 | frozen final reporting |

Main experiments use all 26,557,961 impressions without negative downsampling.

Point-in-time rules:

- joins use raw IDs;
- encoding occurs only after joins;
- encoders and scalers are fitted on training data only;
- behavior features use completed dates before the impression date;
- current and future labels are never used as features;
- identity checks verify that histories belong to the correct raw user.

## Current Pipeline

Run from the repository root:

```bash
python src/01_validate_raw.py
python src/02_profile_raw.py
python src/03_build_base_tables.py
python src/04_build_daily_behavior.py
python src/05_build_behavior_features.py
python src/05b_check_behavior_identity.py
python src/06_profile_model_features.py
python src/07_build_model_inputs.py
python src/08_train_logistic_regression.py
python src/09_diagnose_lr_calibration.py
```

| Step | Purpose |
|---|---|
| 01 | validate raw schemas, labels, keys, and timestamps |
| 02 | profile missing values, prices, and behavior dates |
| 03 | join static tables and create Beijing-time splits |
| 04 | aggregate behavior by raw user and Beijing date |
| 05 | construct leakage-free 1/3/7/14-day behavior features |
| 05b | verify user identity in behavior features |
| 06 | define and profile the model feature schema |
| 07 | build train-only encodings and model inputs |
| 08 | train the out-of-core Logistic Regression baseline |
| 09 | diagnose ranking and probability calibration |

## Repository Layout

```text
data/                 local data; ignored by Git
src/                  pipeline and training entry points
artifacts/            encoders, models, and predictions
results/              small metric and validation reports
docs/                 experimental protocol
requirements.txt      implemented dependencies
```

Large datasets, mapping tables, trained models, and predictions are not committed. Small manifests, metrics, and diagnostic tables are committed.

## Information Layers

- **F1 — Static/context:** user, ad, category, campaign, advertiser, brand, placement, user profile, price, hour, and validated calendar features.
- **F2 — Historical behavior:** leakage-free page-view, favorite, cart, purchase, and recency features over 1/3/7/14-day windows.
- **F3 — Past-only click feedback:** smoothed historical click statistics using only dates before the prediction date.
- **F4 — Behavior sequence:** timestamp-valid category, brand, action, and time-gap histories.

## Planned Models

- Logistic Regression
- LightGBM
- Wide & Deep
- DeepFM
- DIN

LR and LightGBM measure F1/F2/F3 information gain. LR, LightGBM, Wide & Deep, DeepFM, and a sequence-free DIN backbone are compared under a common observable-information budget. DIN sequence variants are evaluated separately because they receive additional sequential information.

## Evaluation

Core evaluation includes LogLoss, AUC, GAUC, PR-AUC, COPC, reliability curves, cold-start segments, model size, and inference throughput.

Post-training Platt scaling and isotonic regression are fitted on validation predictions and evaluated on the frozen test date. Calibration improvement is reported separately from ranking improvement.

See [`docs/EXPERIMENTAL_PROTOCOL.md`](docs/EXPERIMENTAL_PROTOCOL.md) for the fixed experimental plan.

## Current Status

The point-in-time preprocessing and common tabular model-input pipeline are complete. The LR baseline has been stabilized: the discrete weekday feature was removed because the validation weekday was absent from training. The next phase builds the shared evaluator, followed by past-only CTR features.

Previous outputs are preserved on the `archive/v1-original` branch and are not used as evidence for this rebuilt pipeline.
