"""Train Logistic Regression on Version A of the Alimama wide table.

LR is the primary categorical baseline. Low/medium-cardinality features are
one-hot encoded; high-cardinality IDs (user, adgroup_id, campaign_id, customer)
are represented by their CTR aggregate features instead, which avoids a
2M+-dimensional sparse matrix and severe overfitting.

Unlike DIN/DeepFM, LR has no embeddings, so CTR aggregates do not trigger
the within-train embedding-leakage problem — they are included here as signal.

Run: `python scripts/train_lr.py`
"""

# %% Setup
import polars as pl
import numpy as np
from pathlib import Path
import time
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
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
# One-hot: low/medium-cardinality categoricals only (total OHE dims ~9K).
# user (1.1M), adgroup_id (846K), campaign_id (423K), customer (255K) excluded —
# their identity signal is captured by the CTR aggregates in DENSE_FEATURES.
OHE_FEATURES = [
    'pid', 'cate_id', 'cms_segid', 'cms_group_id', 'final_gender_code',
    'age_level', 'pvalue_level', 'shopping_level', 'occupation',
    'new_user_class_level', 'has_profile', 'hour', 'day_of_week',
]

DENSE_FEATURES = [
    'price', 'log_price',
    'user_imp_count', 'user_ctr',
    'ad_imp_count',   'ad_ctr',
    'cate_ctr', 'user_cate_ctr',
    'user_total_events', 'user_total_pv', 'user_total_buy',
    'user_total_cart', 'user_total_fav', 'user_n_unique_cates', 'user_buy_rate',
]

print(f'{len(OHE_FEATURES)} OHE categoricals + {len(DENSE_FEATURES)} dense features')
for col in OHE_FEATURES:
    print(f'  {col:25s}: {trainA[col].n_unique():,} unique values')


# %% Build feature matrices
def toOheArray(df, cols):
    """Extract OHE columns from polars df as a 2D int32 numpy array."""
    return np.column_stack([df[c].to_numpy().astype(np.int32) for c in cols])

def toDenseArray(df, cols):
    return np.column_stack([df[c].to_numpy().astype(np.float32) for c in cols])

print('\nFitting OHE on train...')
t0 = time.time()
ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore', dtype=np.float32)
ohe.fit(toOheArray(trainA, OHE_FEATURES))
oheDims = sum(len(c) for c in ohe.categories_)
print(f'  OHE dims: {oheDims:,}  ({time.time()-t0:.1f}s)')

print('Fitting scaler on train dense features...')
scaler = StandardScaler()
scaler.fit(toDenseArray(trainA, DENSE_FEATURES))

def buildMatrix(df, label=''):
    t0 = time.time()
    ohePart   = ohe.transform(toOheArray(df, OHE_FEATURES))
    densePart = csr_matrix(scaler.transform(toDenseArray(df, DENSE_FEATURES)).astype(np.float32))
    X = hstack([ohePart, densePart], format='csr')
    print(f'  {label}: {X.shape}, {time.time()-t0:.1f}s')
    return X

print('Building feature matrices...')
trainX = buildMatrix(trainA, 'trainX')
valX   = buildMatrix(valA,   'valX')
testX  = buildMatrix(testA,  'testX')

trainY = trainA['clk'].to_numpy().astype(np.float32)
valY   = valA['clk'].to_numpy().astype(np.float32)
testY  = testA['clk'].to_numpy().astype(np.float32)


# %% Train SGDClassifier (logistic regression via SGD)
# SGDClassifier is used over sklearn's LogisticRegression for scalability
# on 20M rows. loss='log_loss' gives exactly logistic regression.
# alpha=1e-6 is weak L2 — CTR datasets are dense in signal; strong reg hurts.
model = SGDClassifier(
    loss='log_loss',
    penalty='l2',
    alpha=1e-6,
    max_iter=3,
    tol=1e-4,
    random_state=42,
    verbose=1,
)

print(f'\nTraining SGDClassifier (logistic regression)...')
t0 = time.time()
model.fit(trainX, trainY)
print(f'Training done in {(time.time()-t0)/60:.1f} min')


# %% Evaluate
print('\nEvaluating on val...')
valPreds = model.predict_proba(valX)[:, 1]
valAuc = roc_auc_score(valY, valPreds)
print(f'Val AUC: {valAuc:.4f}')

print('Evaluating on test...')
testPreds = model.predict_proba(testX)[:, 1]
testAuc   = roc_auc_score(testY, testPreds)
testLoss  = log_loss(testY, testPreds)

print(f'\nTest AUC:      {testAuc:.4f}')
print(f'Test Logloss:  {testLoss:.4f}')


# %% Save predictions for cross-model analysis
PREDS_DIR = Path('../checkpoints') if Path('..').name == 'scripts' else Path('checkpoints')
PREDS_DIR.mkdir(parents=True, exist_ok=True)
np.save(PREDS_DIR / 'preds_lr.npy', testPreds)
np.save(PREDS_DIR / 'test_labels.npy', testY)
print(f'Predictions saved → {PREDS_DIR / "preds_lr.npy"}')
