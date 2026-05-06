"""Train DIN on Version A of the Alimama wide table.

Run interactively in VS Code with Shift+Enter on each `# %%` cell,
or run the whole file: `python scripts/train_din.py`.
"""

# %% Setup
import polars as pl
import numpy as np
import torch
from pathlib import Path
import time

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

DENSE_FEATURES = [
    'price', 'log_price',
    # CTR aggregates excluded from DIN: they are constant per-user/per-ad in train
    # and combined with high-cardinality embeddings cause severe within-train
    # label leakage (train AUC > 0.9, val AUC < 0.6). Baseline models keep them.
    'user_imp_count',
    'ad_imp_count',
    'user_total_events', 'user_total_pv', 'user_total_buy',
    'user_total_cart', 'user_total_fav', 'user_n_unique_cates', 'user_buy_rate',
]

# Vocab size = max + 1 across all splits (must cover val/test categories too)
vocabSizes = {}
for col in SPARSE_FEATURES:
    maxVal = max(trainA[col].max(), valA[col].max(), testA[col].max())
    vocabSizes[col] = int(maxVal) + 1

print(f'{len(SPARSE_FEATURES)} sparse + {len(DENSE_FEATURES)} dense + 1 sequence')
print('\nVocab sizes:')
for col, v in vocabSizes.items():
    print(f'  {col:25s}: {v:>10,}')


# %% Convert behavior_seq from List[Int16] to 2D numpy
def seqToNumpy(df, K=50):
    """Polars List column -> 2D numpy via explode + reshape."""
    n = len(df)
    return df['behavior_seq'].explode().to_numpy().reshape(n, K).astype(np.int64)

print('Converting behavior_seq to 2D numpy...')

t0 = time.time()
trainA_seq = seqToNumpy(trainA)
print(f'  trainA_seq: {trainA_seq.shape}, {(time.time()-t0):.1f}s')

t0 = time.time()
valA_seq = seqToNumpy(valA)
print(f'  valA_seq:   {valA_seq.shape}, {(time.time()-t0):.1f}s')

t0 = time.time()
testA_seq = seqToNumpy(testA)
print(f'  testA_seq:  {testA_seq.shape}, {(time.time()-t0):.1f}s')

print(f'\nFirst row last 10: {trainA_seq[0][-10:]}')


# %% Build input dicts (DeepCTR-Torch format)
def buildInputDict(df, seqArray):
    """DeepCTR-Torch DIN expects a dict {feature_name: np.array}.

    Note: all arrays are float32 (including sparse IDs) to avoid DeepCTR-Torch's
    internal np.concatenate producing float64 tensors -- MPS does not support float64.
    Sparse IDs are stored as float32 but cast to long internally for embedding lookups;
    this is safe because all our IDs fit within float32 precision (max < 2^24).
    """
    d = {}
    for col in SPARSE_FEATURES:
        d[col] = df[col].to_numpy().astype(np.float32)
    for col in DENSE_FEATURES:
        d[col] = df[col].to_numpy().astype(np.float32)
    d['hist_cate_id'] = seqArray.astype(np.float32)
    d['seq_length'] = (seqArray != 0).sum(axis=1).astype(np.float32)
    return d

print('Building input dicts...')
trainA_input = buildInputDict(trainA, trainA_seq)
valA_input   = buildInputDict(valA, valA_seq)
testA_input  = buildInputDict(testA, testA_seq)

trainA_y = trainA['clk'].to_numpy().astype(np.float32)
valA_y   = valA['clk'].to_numpy().astype(np.float32)
testA_y  = testA['clk'].to_numpy().astype(np.float32)

print(f'\nTrain: {len(trainA_y):,} | Val: {len(valA_y):,} | Test: {len(testA_y):,}')
print('\nSample input shapes:')
print(f'  user:         {trainA_input["user"].shape},   dtype={trainA_input["user"].dtype}')
print(f'  hist_cate_id: {trainA_input["hist_cate_id"].shape}, dtype={trainA_input["hist_cate_id"].dtype}')
print(f'  seq_length:   {trainA_input["seq_length"].shape},   dtype={trainA_input["seq_length"].dtype}')
print(f'  price:        {trainA_input["price"].shape},   dtype={trainA_input["price"].dtype}')


# %% Build DIN model (DeepCTR-Torch)
import os
# Allow PyTorch MPS to fall back to CPU for ops MPS doesn't support yet.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

from deepctr_torch.models import DIN
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat

EMBED_DIM = 16  # Run 3: restored to 16 (capacity not the problem; CTR leakage was)

featureColumns = [
    SparseFeat(col, vocabulary_size=vocabSizes[col], embedding_dim=EMBED_DIM)
    for col in SPARSE_FEATURES
]
featureColumns += [DenseFeat(col, 1) for col in DENSE_FEATURES]

# Behavior sequence shares embedding with the candidate cate_id (embedding_name='cate_id').
featureColumns += [
    VarLenSparseFeat(
        SparseFeat('hist_cate_id', vocabulary_size=vocabSizes['cate_id'],
                   embedding_dim=EMBED_DIM, embedding_name='cate_id'),
        maxlen=50, length_name='seq_length',
    )
]

# DIN's history_feature_list names which sparse features have a paired history sequence.
behaviorFeatureList = ['cate_id']

model = DIN(
    dnn_feature_columns=featureColumns,
    history_feature_list=behaviorFeatureList,
    dnn_hidden_units=(256, 128),
    dnn_dropout=0.5,
    task='binary',
    device=device,
)

model.compile(
    optimizer='adam',  # default lr=1e-3
    loss='binary_crossentropy',
    metrics=['binary_crossentropy', 'auc'],
)

print(f'Model created on device={device}')
nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters: {nParams:,}')


# %% Smoke test: train 1 epoch on a small subset to verify the pipeline runs
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
print(f'Smoke test done in {time.time()-t0:.1f}s. If you see an AUC line above, pipeline works.')


# %% Full training with early stopping on val
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint

CHECKPOINT_PATH = '../checkpoints/din_versionA_run3.pt' if Path('..').name == 'scripts' else 'checkpoints/din_versionA_run3.pt'
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


# %% Save predictions for cross-model analysis
PREDS_DIR = Path(CHECKPOINT_PATH).parent
PREDS_DIR.mkdir(parents=True, exist_ok=True)
np.save(PREDS_DIR / 'preds_din.npy', testPreds)
np.save(PREDS_DIR / 'test_labels.npy', testA_y)
print(f'Predictions saved → {PREDS_DIR / "preds_din.npy"}')
