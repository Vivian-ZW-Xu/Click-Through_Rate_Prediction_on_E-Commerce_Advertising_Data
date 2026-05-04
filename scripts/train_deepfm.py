"""Train DeepFM on Version A of the Alimama wide table.

DeepFM = FM (auto 2-way feature crossing) + Deep component (MLP).
No behavior sequence input (DeepFM is a static-feature model).

Run: `python scripts/train_deepfm.py` from project root.
"""

# %% Setup
import polars as pl
import numpy as np
import torch
from pathlib import Path
import time
import os

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

DATA_DIR = Path('../data/processed') if Path('../data/processed').exists() else Path('data/processed')
WIDE = DATA_DIR / 'wide'

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Device: {device}')


# %% Load Version A
print('Loading...')
trainA = pl.read_parquet(WIDE / 'trainA.parquet')
valA   = pl.read_parquet(WIDE / 'valA.parquet')
testA  = pl.read_parquet(WIDE / 'testA.parquet')

print(f'trainA: {trainA.shape}')
print(f'valA:   {valA.shape}')
print(f'testA:  {testA.shape}')


# %% Feature lists + vocab sizes
SPARSE_FEATURES = [
    'user', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer',
    'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
    'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
    'has_profile', 'hour', 'day_of_week',
]

# Same rationale as DIN: CTR aggregates cause within-train leakage with
# high-cardinality embeddings. Keep counts and behavior_log aggregates.
DENSE_FEATURES = [
    'price', 'log_price',
    'user_imp_count',
    'ad_imp_count',
    'user_total_events', 'user_total_pv', 'user_total_buy',
    'user_total_cart', 'user_total_fav', 'user_n_unique_cates', 'user_buy_rate',
]

vocabSizes = {}
for col in SPARSE_FEATURES:
    maxVal = max(trainA[col].max(), valA[col].max(), testA[col].max())
    vocabSizes[col] = int(maxVal) + 1

print(f'{len(SPARSE_FEATURES)} sparse + {len(DENSE_FEATURES)} dense (no sequence for DeepFM)')


# %% Build input dicts (no behavior_seq for DeepFM)
def buildInputDict(df):
    """DeepFM input format: dict {feature_name: np.array}, all float32 for MPS compat."""
    d = {}
    for col in SPARSE_FEATURES:
        d[col] = df[col].to_numpy().astype(np.float32)
    for col in DENSE_FEATURES:
        d[col] = df[col].to_numpy().astype(np.float32)
    return d

print('Building input dicts...')
trainA_input = buildInputDict(trainA)
valA_input   = buildInputDict(valA)
testA_input  = buildInputDict(testA)

trainA_y = trainA['clk'].to_numpy().astype(np.float32)
valA_y   = valA['clk'].to_numpy().astype(np.float32)
testA_y  = testA['clk'].to_numpy().astype(np.float32)

print(f'\nTrain: {len(trainA_y):,} | Val: {len(valA_y):,} | Test: {len(testA_y):,}')


# %% Build DeepFM model
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat

EMBED_DIM = 16

featureColumns = [
    SparseFeat(col, vocabulary_size=vocabSizes[col], embedding_dim=EMBED_DIM)
    for col in SPARSE_FEATURES
]
featureColumns += [DenseFeat(col, 1) for col in DENSE_FEATURES]

# DeepFM combines:
#   - linear part (1st-order)
#   - FM part (2nd-order feature interactions, shared embedding)
#   - DNN part (higher-order via MLP)
model = DeepFM(
    linear_feature_columns=featureColumns,
    dnn_feature_columns=featureColumns,
    dnn_hidden_units=(256, 128),
    dnn_dropout=0.5,
    task='binary',
    device=device,
)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_crossentropy', 'auc'],
)

print(f'Model created on device={device}')
nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters: {nParams:,}')


# %% Smoke test: 1 epoch on 100K rows
SMOKE_N = 100_000

print(f'\nSmoke test on {SMOKE_N:,} rows, 1 epoch...')
smokeInput = {k: v[:SMOKE_N] for k, v in trainA_input.items()}
smokeY = trainA_y[:SMOKE_N]

t0 = time.time()
model.fit(
    x=smokeInput,
    y=smokeY,
    batch_size=2048,
    epochs=1,
    verbose=2,
)
print(f'Smoke test done in {time.time()-t0:.1f}s.')


# %% Full training with early stopping on val
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint

CHECKPOINT_PATH = '../checkpoints/deepfm_versionA.pt' if Path('..').name == 'scripts' else 'checkpoints/deepfm_versionA.pt'
Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)

earlyStop = EarlyStopping(monitor='val_auc', patience=2, mode='max')
checkpoint = ModelCheckpoint(
    filepath=CHECKPOINT_PATH, monitor='val_auc', mode='max',
    save_best_only=True, save_weights_only=True, verbose=1,
)

print('\nFull training on all of trainA...')
t0 = time.time()
history = model.fit(
    x=trainA_input,
    y=trainA_y,
    batch_size=4096,
    epochs=10,
    validation_data=(valA_input, valA_y),
    callbacks=[earlyStop, checkpoint],
    verbose=2,
)
print(f'\nTotal training time: {(time.time()-t0)/60:.1f} min')


# %% Evaluate on test set
from sklearn.metrics import roc_auc_score, log_loss

print('Loading best checkpoint and predicting on test...')
model.load_state_dict(torch.load(CHECKPOINT_PATH))

testPreds = model.predict(testA_input, batch_size=4096)
testAuc = roc_auc_score(testA_y, testPreds)
testLogloss = log_loss(testA_y, testPreds)

print(f'\nTest AUC:      {testAuc:.4f}')
print(f'Test Logloss:  {testLogloss:.4f}')
