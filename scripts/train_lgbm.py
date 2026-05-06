"""Train LightGBM on Version A of the Alimama wide table.

LightGBM handles categorical features natively (Fisher split scoring).
Low/medium-cardinality features are declared as categoricals; high-cardinality
IDs (user, adgroup_id, campaign_id, customer) are passed as raw integer columns
alongside CTR aggregates that carry their identity signal.

CTR aggregates are included here (unlike DIN/DeepFM) because tree models do not
suffer from the embedding × CTR within-train leakage issue.

Run: `python scripts/train_lgbm.py`
"""

# %% Setup
import polars as pl
import numpy as np
import lightgbm as lgb
from pathlib import Path
import time
from sklearn.metrics import roc_auc_score, log_loss

DATA_DIR = Path('../data/processed') if Path('../data/processed').exists() else Path('data/processed')
WIDE = DATA_DIR / 'wide'


# %% Load Version A
print('Loading...')
trainA = pl.read_parquet(WIDE / 'trainA.parquet')
valA   = pl.read_parquet(WIDE / 'valA.parquet')
testA  = pl.read_parquet(WIDE / 'testA.parquet')

print(f'trainA: {trainA.shape}')
print(f'valA:   {valA.shape}')
print(f'testA:  {testA.shape}')


# %% Feature definition
# Low/medium-cardinality → LGB native categorical treatment.
CAT_FEATURES = [
    'pid', 'cate_id', 'cms_segid', 'cms_group_id', 'final_gender_code',
    'age_level', 'pvalue_level', 'shopping_level', 'occupation',
    'new_user_class_level', 'has_profile', 'hour', 'day_of_week',
]

# High-cardinality IDs as raw integers; LGB splits on them ordinally,
# complemented by CTR aggregates below.
HIGH_CARD_FEATURES = ['user', 'adgroup_id', 'campaign_id', 'customer']

DENSE_FEATURES = [
    'price', 'log_price',
    'user_imp_count', 'user_ctr',
    'ad_imp_count',   'ad_ctr',
    'cate_ctr', 'user_cate_ctr',
    'user_total_events', 'user_total_pv', 'user_total_buy',
    'user_total_cart', 'user_total_fav', 'user_n_unique_cates', 'user_buy_rate',
]

ALL_FEATURES = CAT_FEATURES + HIGH_CARD_FEATURES + DENSE_FEATURES
# Categorical feature indices correspond to their position in ALL_FEATURES
CAT_INDICES = list(range(len(CAT_FEATURES)))

print(f'{len(CAT_FEATURES)} categorical + {len(HIGH_CARD_FEATURES)} high-card int + {len(DENSE_FEATURES)} dense')
print(f'Total: {len(ALL_FEATURES)} features')


# %% Convert to numpy
print('\nConverting to numpy...')
t0 = time.time()

def toMatrix(df, cols):
    return np.column_stack([df[c].to_numpy() for c in cols])

trainX = toMatrix(trainA, ALL_FEATURES).astype(np.float64)
valX   = toMatrix(valA,   ALL_FEATURES).astype(np.float64)
testX  = toMatrix(testA,  ALL_FEATURES).astype(np.float64)

trainY = trainA['clk'].to_numpy().astype(np.int32)
valY   = valA['clk'].to_numpy().astype(np.int32)
testY  = testA['clk'].to_numpy().astype(np.int32)

print(f'Done in {time.time()-t0:.1f}s')
print(f'trainX: {trainX.shape}  |  valX: {valX.shape}  |  testX: {testX.shape}')


# %% Build LightGBM Datasets
dtrain = lgb.Dataset(
    trainX, label=trainY,
    feature_name=ALL_FEATURES,
    categorical_feature=CAT_INDICES,
    free_raw_data=False,
)
dval = lgb.Dataset(
    valX, label=valY,
    feature_name=ALL_FEATURES,
    categorical_feature=CAT_INDICES,
    free_raw_data=False,
    reference=dtrain,
)


# %% Train LightGBM
params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 255,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'cat_smooth': 10,       # smoothing for categorical splits
    'verbose': 1,
    'seed': 42,
}

print('\nTraining LightGBM...')
t0 = time.time()

booster = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dval],
    valid_names=['val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ],
)

print(f'\nTotal training time: {(time.time()-t0)/60:.1f} min')
print(f'Best iteration: {booster.best_iteration}')

CHECKPOINT_DIR = Path('../checkpoints') if Path('..').name == 'scripts' else Path('checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
booster.save_model(str(CHECKPOINT_DIR / 'lgbm_versionA.txt'))
print(f'Model saved → {CHECKPOINT_DIR / "lgbm_versionA.txt"}')


# %% Evaluate on test set
from sklearn.metrics import roc_auc_score, log_loss

print('\nPredicting on test...')
testPreds = booster.predict(testX, num_iteration=booster.best_iteration)
testAuc  = roc_auc_score(testY, testPreds)
testLoss = log_loss(testY, testPreds)

print(f'\nTest AUC:      {testAuc:.4f}')
print(f'Test Logloss:  {testLoss:.4f}')


# %% Save predictions for cross-model analysis
PREDS_DIR = Path('../checkpoints') if Path('..').name == 'scripts' else Path('checkpoints')
PREDS_DIR.mkdir(parents=True, exist_ok=True)
np.save(PREDS_DIR / 'preds_lgbm.npy', testPreds)
np.save(PREDS_DIR / 'test_labels.npy', testY)
print(f'Predictions saved → {PREDS_DIR / "preds_lgbm.npy"}')
