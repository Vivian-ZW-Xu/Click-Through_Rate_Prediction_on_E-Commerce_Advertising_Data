"""
Cross-model analysis — Alimama CTR project.

Teacher requirements (4 directions):

  DIRECTION 1  Datapoint-Level Agreement Analysis
               ▸ Required models: ALL available (uses allPreds dict)
               ▸ Pairwise agreement matrix, per-sample difficulty score,
                 recall/precision by agreement level, easy vs hard clicks

  DIRECTION 2  Failure Mode Analysis
               ▸ Required models: ALL (calibration curves)
                                  best available model (error breakdown)
               ▸ Calibration curves, AUC by feature decile, FNR by segment

  DIRECTION 3  Attention Analysis
               ▸ Required model:  DIN only
               ▸ Checkpoint:      checkpoints/din_versionA_run3.pt
               ▸ Attention weight by sequence position, clicked vs non-clicked

  DIRECTION 4  Feature Importance
               ▸ LightGBM: checkpoints/lgbm_versionA.txt  (gain + SHAP)
               ▸ LR:       checkpoints/lr_model.pkl  (or auto-retrained)
               ▸ Top feature coefficients / importance scores

Additional (from earlier discussion):
  Per-Segment AUC breakdown     ALL available models
  Sample Complexity Curves      LR + LightGBM retrained at 6 fractions

Run: `python scripts/analysis.py`
Missing checkpoints are skipped gracefully with a warning.
"""

# %% ── Imports ────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import polars as pl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
import time

matplotlib.rcParams.update({
    'figure.dpi': 150, 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})

DATA_DIR = Path('data/processed')  if Path('data/processed').exists()  else Path('../data/processed')
WIDE     = DATA_DIR / 'wide'
CKPT     = Path('checkpoints')     if Path('checkpoints').exists()     else Path('../checkpoints')
FIG_DIR  = Path('figures/analysis')
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ['LR', 'LightGBM', 'Wide&Deep', 'DeepFM', 'DIN']
PRED_FILES  = ['preds_lr', 'preds_lgbm', 'preds_wide_deep', 'preds_deepfm', 'preds_din']
COLORS = {
    'LR': '#888888', 'LightGBM': '#2ca02c', 'Wide&Deep': '#ff7f0e',
    'DeepFM': '#1f77b4', 'DIN': '#d62728',
}


# %% ── Load predictions ───────────────────────────────────────────────────────
print('Loading predictions...')
testY = np.load(CKPT / 'test_labels.npy')
print(f'Test set: {len(testY):,} samples  CTR={testY.mean():.4f}')

allPreds = {}
for name, fname in zip(MODEL_NAMES, PRED_FILES):
    path = CKPT / f'{fname}.npy'
    if path.exists():
        allPreds[name] = np.load(path).ravel()   # ravel: handle (N,1) saves from deepctr-torch
        print(f'  {name:12s}: AUC={roc_auc_score(testY, allPreds[name]):.4f}')
    else:
        print(f'  {name:12s}: not found — skipped')

MODELS = list(allPreds.keys())
if not MODELS:
    raise SystemExit('No prediction files found. Run training scripts first.')
print(f'\n{len(MODELS)} model(s) loaded: {MODELS}')

# Determine best available model by test AUC for failure mode analysis
bestModel = max(MODELS, key=lambda m: roc_auc_score(testY, allPreds[m]))
print(f'Best model for failure mode analysis: {bestModel}')


# %% ── Load data ──────────────────────────────────────────────────────────────
# Load testA WITHOUT behavior_seq (2.8M × 50 int16 ≈ 280 MB saved).
# behavior_seq is only needed for Direction 3 (DIN attention) and loaded there
# lazily as a small subset. valA is only needed for Sample Complexity and
# loaded there to avoid keeping 3 M-row dataframes in RAM simultaneously.
print('\nLoading testA (scalar columns only)...')
testA = (
    pl.scan_parquet(WIDE / 'testA.parquet')
    .drop('behavior_seq')
    .collect()
)
print(f'testA: {testA.shape}')


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTION 1 — DATAPOINT-LEVEL AGREEMENT ANALYSIS
# Required models: ALL available models (minimum 2 recommended)
# Answers: which clicks do all/some/no models catch? How similar are models?
# ═══════════════════════════════════════════════════════════════════════════════

# %% D1 — Binarise predictions at top-10% threshold (≈ 2× global CTR ~5%)
print('\n' + '='*60)
print('DIRECTION 1: Datapoint-Level Agreement Analysis')
print('='*60)

TOPK = 0.10
binaryPreds = {
    m: (allPreds[m] >= np.quantile(allPreds[m], 1 - TOPK)).astype(np.int8)
    for m in MODELS
}
agreementScore = sum(binaryPreds.values())   # shape (N,): 0..len(MODELS)
nM = len(MODELS)
totalClicks = int(testY.sum())
clickMask   = testY == 1


# %% D1a — Pairwise agreement matrix on actual clicks
# Entry (i,j) = fraction of actual clicks that BOTH model i and model j catch.
# Diagonal = individual recall. Off-diagonal = joint recall.
pairMat = np.zeros((nM, nM))
for i, m1 in enumerate(MODELS):
    for j, m2 in enumerate(MODELS):
        both = (binaryPreds[m1][clickMask] == 1) & (binaryPreds[m2][clickMask] == 1)
        pairMat[i, j] = both.sum() / totalClicks

print('\nPairwise agreement on actual clicks (fraction of clicks caught by BOTH models):')
print(f'{"":>12}', end='')
for m in MODELS:
    print(f'  {m:>10}', end='')
print()
for i, m in enumerate(MODELS):
    print(f'{m:>12}', end='')
    for j in range(nM):
        print(f'  {pairMat[i,j]:>10.3f}', end='')
    print()

fig, ax = plt.subplots(figsize=(max(5, nM * 1.4), max(4, nM * 1.2)))
im = ax.imshow(pairMat, cmap='Blues', vmin=0, vmax=pairMat.max())
ax.set_xticks(range(nM)); ax.set_xticklabels(MODELS, rotation=30, ha='right')
ax.set_yticks(range(nM)); ax.set_yticklabels(MODELS)
ax.set_title('Pairwise recall on actual clicks\n(fraction of clicks caught by both models)')
for i in range(nM):
    for j in range(nM):
        ax.text(j, i, f'{pairMat[i,j]:.3f}', ha='center', va='center',
                fontsize=9, color='white' if pairMat[i,j] > 0.5*pairMat.max() else 'black')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(FIG_DIR / 'D1a_pairwise_agreement.png', bbox_inches='tight')
plt.show()
print('Saved → D1a_pairwise_agreement.png')


# %% D1b — Per-sample difficulty score (prediction variance across models)
# High variance = models strongly disagree = uncertain / borderline samples.
predsMatrix = np.column_stack([allPreds[m] for m in MODELS])   # (N, n_models)
difficultyScore = predsMatrix.std(axis=1)                        # (N,)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.hist(difficultyScore[testY == 0], bins=50, alpha=0.6, density=True,
        color='steelblue', label='Non-click')
ax.hist(difficultyScore[testY == 1], bins=50, alpha=0.6, density=True,
        color='#d62728', label='Click')
ax.set_xlabel('Prediction std across models (difficulty score)')
ax.set_ylabel('Density')
ax.set_title('Difficulty score: clicks vs non-clicks')
ax.legend()

ax = axes[1]
# AUC of the ensemble (mean prediction) vs difficulty percentile bucket
buckets = np.percentile(difficultyScore, np.linspace(0, 100, 11))
bucket_auc = []
bucket_mid = []
for lo, hi in zip(buckets[:-1], buckets[1:]):
    mask = (difficultyScore >= lo) & (difficultyScore < hi)
    if testY[mask].sum() > 10:
        bucket_auc.append(roc_auc_score(testY[mask], predsMatrix[mask].mean(axis=1)))
        bucket_mid.append((lo + hi) / 2)
ax.plot(bucket_mid, bucket_auc, marker='o', color='#ff7f0e')
ax.set_xlabel('Difficulty score (prediction std) — decile bucket')
ax.set_ylabel('Ensemble AUC within bucket')
ax.set_title('Ensemble AUC by sample difficulty')

plt.suptitle('Per-Sample Difficulty Score Analysis', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'D1b_difficulty_score.png', bbox_inches='tight')
plt.show()
print('Saved → D1b_difficulty_score.png')


# %% D1c — Agreement-level recall and precision
print(f'\nAgreement analysis (top-{TOPK*100:.0f}% threshold, {nM} models):')
print(f'{"Agreement":>12}  {"N samples":>10}  {"Clicks":>8}  {"Recall%":>8}  {"Precision":>10}')
print('-' * 58)
for n in range(nM + 1):
    mask = agreementScore == n
    n_s  = int(mask.sum())
    clks = int(testY[mask].sum())
    rec  = 100 * clks / totalClicks
    prec = testY[mask].mean() if n_s > 0 else 0
    print(f'  {n} model(s):  {n_s:>10,}  {clks:>8,}  {rec:>7.1f}%  {prec:>10.4f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
levels = list(range(nM + 1))
n_clks = [(testY[agreementScore == n] == 1).sum() for n in levels]
n_neg  = [(testY[agreementScore == n] == 0).sum() for n in levels]
ctrs   = [testY[agreementScore == n].mean() if (agreementScore == n).any() else 0 for n in levels]

axes[0].bar(levels, n_neg,  label='Non-click', color='#aec7e8')
axes[0].bar(levels, n_clks, label='Click', color='#d62728', bottom=n_neg)
axes[0].set_yscale('log'); axes[0].set_xlabel(f'Models predicting positive (of {nM})')
axes[0].set_ylabel('Count (log)'); axes[0].set_title('Sample distribution by agreement level')
axes[0].legend()

axes[1].bar(levels, ctrs, color='#ff7f0e')
axes[1].axhline(testY.mean(), color='gray', linestyle='--',
                label=f'Global CTR ({testY.mean():.3f})')
axes[1].set_xlabel(f'Models predicting positive (of {nM})')
axes[1].set_ylabel('CTR (precision)'); axes[1].set_title('Precision by agreement level')
axes[1].legend()

plt.suptitle('D1c — Agreement-Level Recall & Precision', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'D1c_agreement_recall.png', bbox_inches='tight')
plt.show()
print('Saved → D1c_agreement_recall.png')


# %% D1d — Feature distributions: easy vs hard clicks
easyMask = (testY == 1) & (agreementScore == nM)   # all models catch
hardMask = (testY == 1) & (agreementScore == 0)    # no model catches
print(f'\nEasy clicks (all {nM} models): {easyMask.sum():,}')
print(f'Hard clicks (no model):         {hardMask.sum():,}')

compareFeats = ['user_imp_count', 'user_ctr', 'ad_ctr', 'cate_ctr',
                'user_cate_ctr', 'user_total_events', 'user_buy_rate', 'price']

print(f'\n{"Feature":<22}  {"Easy":>10}  {"Hard":>10}  {"All clicks":>12}')
print('-' * 60)
for col in compareFeats:
    arr = testA[col].to_numpy().astype(float)
    e = arr[easyMask].mean() if easyMask.sum() > 0 else float('nan')
    h = arr[hardMask].mean() if hardMask.sum() > 0 else float('nan')
    a = arr[clickMask].mean()
    print(f'{col:<22}  {e:>10.4f}  {h:>10.4f}  {a:>12.4f}')

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, col in zip(axes.flatten(), compareFeats):
    arr = testA[col].to_numpy().astype(float)
    cap = np.percentile(arr[clickMask], 98)
    bins = np.linspace(0, cap, 30)
    if easyMask.sum() > 0:
        ax.hist(np.clip(arr[easyMask], 0, cap), bins=bins, alpha=0.6,
                density=True, color='#2ca02c', label=f'Easy (all {nM})')
    if hardMask.sum() > 0:
        ax.hist(np.clip(arr[hardMask], 0, cap), bins=bins, alpha=0.6,
                density=True, color='#d62728', label='Hard (none)')
    ax.set_title(col, fontsize=9); ax.set_yticks([])
    if col == compareFeats[0]: ax.legend(fontsize=8)

plt.suptitle('D1d — Feature Distributions: Easy vs Hard Clicks', fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / 'D1d_easy_vs_hard.png', bbox_inches='tight')
plt.show()
print('Saved → D1d_easy_vs_hard.png')


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTION 2 — FAILURE MODE ANALYSIS
# Required models: ALL (calibration curves)
#                  bestModel (error breakdown by feature + FNR by segment)
# Answers: are predictions well-calibrated? where does the best model fail most?
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('DIRECTION 2: Failure Mode Analysis')
print(f'Best model: {bestModel}')
print('='*60)


# %% D2a — Calibration curves (reliability diagrams) for ALL models
from sklearn.calibration import calibration_curve

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1.5)
for m in MODELS:
    frac_pos, mean_pred = calibration_curve(
        testY, allPreds[m], n_bins=20, strategy='quantile'
    )
    ax.plot(mean_pred, frac_pos, marker='o', markersize=3,
            label=m, color=COLORS.get(m), linewidth=1.5)
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives (true CTR)')
ax.set_title('Calibration curves (reliability diagram)')
ax.set_xlim(0, 0.3)
ax.set_ylim(0, 0.2)
ax.legend(fontsize=9)

# Right: distribution of predicted probabilities per model
ax = axes[1]
for m in MODELS:
    ax.hist(allPreds[m], bins=50, alpha=0.5, density=True,
            label=m, color=COLORS.get(m), range=(0, 0.3))
ax.axvline(testY.mean(), color='black', linestyle='--',
           label=f'True CTR ({testY.mean():.3f})', linewidth=1.5)
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Density')
ax.set_title('Predicted score distributions')
ax.set_xlim(0, 0.3)
ax.legend(fontsize=9)

plt.suptitle('D2a — Model Calibration', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'D2a_calibration.png', bbox_inches='tight')
plt.show()
print('Saved → D2a_calibration.png')


# %% D2b — AUC by decile of key features (best model only)
# Reveals which feature ranges the best model handles worst.
auditFeats = ['user_ctr', 'user_cate_ctr', 'ad_ctr', 'user_imp_count']
N_BINS = 10
Y_LIM = (0.45, 0.80)   # unified y-axis across subplots for fair comparison

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
axes = axes.flatten()

overallAuc = roc_auc_score(testY, allPreds[bestModel])

for ax, col in zip(axes, auditFeats):
    vals = testA[col].to_numpy().astype(float)
    edges = np.nanpercentile(vals, np.linspace(0, 100, N_BINS + 1))
    edges = np.unique(edges)
    bin_aucs, bin_mids, bin_sizes = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (vals >= lo) & (vals <= hi)
        if testY[mask].sum() >= 10:
            bin_aucs.append(roc_auc_score(testY[mask], allPreds[bestModel][mask]))
            bin_mids.append((lo + hi) / 2)
            bin_sizes.append(mask.sum())
    ax.plot(bin_mids, bin_aucs, marker='o', color=COLORS.get(bestModel, 'steelblue'))
    ax.axhline(overallAuc, color='gray', linestyle='--', linewidth=1,
               label=f'Overall AUC ({overallAuc:.3f})')
    ax.set_xlabel(col); ax.set_title(col)
    ax.set_ylim(*Y_LIM)
    ax.legend(fontsize=8, loc='lower right')

axes[0].set_ylabel('AUC')
axes[2].set_ylabel('AUC')
plt.suptitle(f'D2b — {bestModel}: AUC by Feature Decile', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'D2b_auc_by_decile.png', bbox_inches='tight')
plt.show()
print('Saved → D2b_auc_by_decile.png')


# %% D2c — False Negative Rate (missed clicks) by user activity segment
segments_fnr = {
    'Cold user\n(imp<10)':   (testA['user_imp_count'] <  10).to_numpy(),
    'Warm user\n(10-100)':   (testA['user_imp_count'].is_between(10, 100)).to_numpy(),
    'Hot user\n(imp>100)':   (testA['user_imp_count'] > 100).to_numpy(),
    'No profile':            (testA['has_profile'] == 0).to_numpy(),
    'Has profile':           (testA['has_profile'] == 1).to_numpy(),
    'Low behavior\n(<100ev)':(testA['user_total_events'] < 100).to_numpy(),
    'High behavior\n(≥500ev)':(testA['user_total_events'] >= 500).to_numpy(),
}

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(segments_fnr))
width = 0.75 / nM

for i, m in enumerate(MODELS):
    # threshold at top-10% for this model
    thresh = np.quantile(allPreds[m], 1 - TOPK)
    predicted_pos = allPreds[m] >= thresh
    fnrs = []
    for mask in segments_fnr.values():
        actual_clicks = (testY == 1) & mask
        missed = actual_clicks & ~predicted_pos
        fnrs.append(missed.sum() / actual_clicks.sum() if actual_clicks.sum() > 0 else 0)
    offset = i * width - (nM - 1) * width / 2
    ax.bar(x + offset, fnrs, width, label=m, color=COLORS.get(m))

ax.set_xticks(x)
ax.set_xticklabels(list(segments_fnr.keys()), fontsize=9)
ax.set_ylabel('False Negative Rate (missed clicks / actual clicks)')
ax.set_title('D2c — False Negative Rate by User Segment')
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'D2c_fnr_by_segment.png', bbox_inches='tight')
plt.show()
print('Saved → D2c_fnr_by_segment.png')


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTION 3 — ATTENTION ANALYSIS
# Required model: DIN only
# Checkpoint:     checkpoints/din_versionA_run3.pt
# Answers: does DIN attend to recent history? do clicked impressions have
#          higher / more focused attention than non-clicked ones?
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('DIRECTION 3: Attention Analysis (DIN only)')
print('='*60)

DIN_CKPT = CKPT / 'din_versionA_run3.pt'

if not DIN_CKPT.exists():
    print(f'  Skipped — {DIN_CKPT} not found. Run train_din.py first.')
else:
    import torch, os
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    from deepctr_torch.models import DIN
    from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'  Device: {device}')

    # %% D3 — Reconstruct DIN model (must match train_din.py exactly)
    print('  Loading trainA to compute vocab sizes...')
    trainA_din = pl.read_parquet(WIDE / 'trainA.parquet')

    SPARSE_FEATS_DIN = [
        'user', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer',
        'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
        'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
        'has_profile', 'hour', 'day_of_week',
    ]
    DENSE_FEATS_DIN = [
        'price', 'log_price', 'user_imp_count', 'ad_imp_count',
        'user_total_events', 'user_total_pv', 'user_total_buy',
        'user_total_cart', 'user_total_fav', 'user_n_unique_cates', 'user_buy_rate',
    ]
    EMBED_DIM = 16

    # valA not loaded globally to save RAM; vocab max from trainA+testA is sufficient
    # (val IDs are a strict subset of train IDs after label-encoding in preprocessing)
    vocabSizes = {}
    for col in SPARSE_FEATS_DIN:
        vocabSizes[col] = int(max(
            trainA_din[col].max(), testA[col].max()
        )) + 1

    featureCols = [
        SparseFeat(c, vocabulary_size=vocabSizes[c], embedding_dim=EMBED_DIM)
        for c in SPARSE_FEATS_DIN
    ] + [DenseFeat(c, 1) for c in DENSE_FEATS_DIN] + [
        VarLenSparseFeat(
            SparseFeat('hist_cate_id', vocabulary_size=vocabSizes['cate_id'],
                       embedding_dim=EMBED_DIM, embedding_name='cate_id'),
            maxlen=50, length_name='seq_length',
        )
    ]

    din_model = DIN(
        dnn_feature_columns=featureCols,
        history_feature_list=['cate_id'],
        dnn_hidden_units=(256, 128), dnn_dropout=0.5,
        task='binary', device=device,
    )
    din_model.load_state_dict(torch.load(DIN_CKPT, map_location=device))
    din_model.eval()
    print('  DIN checkpoint loaded.')

    # %% D3 — Register forward hook on LocalActivationUnit to capture attention
    attn_store = []
    def _attn_hook(_module, _inp, out):
        w = out.detach().cpu()
        # Normalise shape to (batch, seq_len)
        while w.dim() > 2:
            w = w.squeeze(-1) if w.shape[-1] == 1 else w.squeeze(1)
        attn_store.append(w.numpy())

    hooks = []
    for name, mod in din_model.named_modules():
        if 'LocalActivationUnit' in type(mod).__name__:
            hooks.append(mod.register_forward_hook(_attn_hook))
            print(f'  Hooked attention layer: {name}')
            break
    if not hooks:
        print('  WARNING: Could not find LocalActivationUnit — attention analysis skipped.')

    if hooks:
        N_ATTN = 5_000
        testY_sub = testY[:N_ATTN]

        # Load behavior_seq for first N_ATTN rows only (lazy scan = memory-safe)
        testA_sub = pl.scan_parquet(WIDE / 'testA.parquet').head(N_ATTN).collect()

        def seqToNumpy(df, K=50):
            return df['behavior_seq'].explode().to_numpy().reshape(len(df), K).astype(np.int64)

        seqArr = seqToNumpy(testA_sub)
        attn_input = {}
        for col in SPARSE_FEATS_DIN:
            attn_input[col] = testA_sub[col].to_numpy().astype(np.float32)
        for col in DENSE_FEATS_DIN:
            attn_input[col] = testA_sub[col].to_numpy().astype(np.float32)
        attn_input['hist_cate_id'] = seqArr.astype(np.float32)
        attn_input['seq_length']   = (seqArr != 0).sum(axis=1).astype(np.float32)

        print(f'  Running DIN forward pass on {N_ATTN:,} samples to extract attention...')
        with torch.no_grad():
            din_model.predict(attn_input, batch_size=512)

        for h in hooks:
            h.remove()

        attn_weights = np.concatenate(attn_store, axis=0)   # (N_ATTN, 50)
        print(f'  Attention weights shape: {attn_weights.shape}')

        # Mask padding positions (cate_id == 0 means padded)
        pad_mask = seqArr == 0                               # True where padded
        attn_weights_masked = np.where(pad_mask, np.nan, attn_weights)

        # %% D3a — Mean attention weight by sequence position
        # Position 0 = oldest behaviour, position 49 = most recent
        mean_by_pos   = np.nanmean(attn_weights_masked, axis=0)
        clicked_attn  = attn_weights_masked[testY_sub == 1]
        nclicked_attn = attn_weights_masked[testY_sub == 0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(range(50), mean_by_pos, color='#d62728', linewidth=2)
        ax.set_xlabel('Sequence position (0=oldest, 49=most recent)')
        ax.set_ylabel('Mean attention weight')
        ax.set_title('D3a — Attention weight by history position\n(Does DIN prefer recent items?)')

        # %% D3b — Clicked vs non-clicked: attention weight distribution
        ax = axes[1]
        mean_clicked  = np.nanmean(clicked_attn,  axis=0)
        mean_nclicked = np.nanmean(nclicked_attn, axis=0)
        ax.plot(range(50), mean_clicked,  color='#d62728', linewidth=2, label='Clicked')
        ax.plot(range(50), mean_nclicked, color='steelblue', linewidth=2, label='Non-clicked')
        ax.set_xlabel('Sequence position (0=oldest, 49=most recent)')
        ax.set_ylabel('Mean attention weight')
        ax.set_title('D3b — Attention: clicked vs non-clicked impressions')
        ax.legend()

        plt.suptitle('DIN Attention Analysis', fontsize=13)
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'D3_attention_analysis.png', bbox_inches='tight')
        plt.show()
        print('Saved → D3_attention_analysis.png')

        # Summary stat
        early_mean = np.nanmean(mean_by_pos[:25])
        late_mean  = np.nanmean(mean_by_pos[25:])
        print(f'\n  Mean attention — oldest 25 positions: {early_mean:.4f}')
        print(f'  Mean attention — newest 25 positions: {late_mean:.4f}')
        print(f'  Recency bias ratio (new/old): {late_mean/early_mean:.2f}x')

        del trainA_din, attn_store, attn_weights


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTION 4 — FEATURE IMPORTANCE
# LightGBM: checkpoints/lgbm_versionA.txt  → gain importance + SHAP
# LR:       checkpoints/lr_model.pkl       → top coefficients
#           (auto-retrained on trainA if pkl not found, ~15 min)
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('DIRECTION 4: Feature Importance')
print('='*60)

# ── Feature definitions (mirror train_lgbm.py and train_lr.py) ───────────────
CAT_FEATS_LGB   = ['pid', 'cate_id', 'cms_segid', 'cms_group_id', 'final_gender_code',
                   'age_level', 'pvalue_level', 'shopping_level', 'occupation',
                   'new_user_class_level', 'has_profile', 'hour', 'day_of_week']
HCARD_FEATS     = ['user', 'adgroup_id', 'campaign_id', 'customer']
DENSE_FEATS_LGB = ['price', 'log_price', 'user_imp_count', 'user_ctr',
                   'ad_imp_count', 'ad_ctr', 'cate_ctr', 'user_cate_ctr',
                   'user_total_events', 'user_total_pv', 'user_total_buy',
                   'user_total_cart', 'user_total_fav', 'user_n_unique_cates', 'user_buy_rate']
ALL_FEATS_LGB   = CAT_FEATS_LGB + HCARD_FEATS + DENSE_FEATS_LGB
CAT_IDX_LGB     = list(range(len(CAT_FEATS_LGB)))

OHE_FEATS       = CAT_FEATS_LGB   # same set used by LR
DENSE_FEATS_LR  = DENSE_FEATS_LGB

def colStack(df, cols):
    return np.column_stack([df[c].to_numpy() for c in cols])


# %% D4a — LightGBM feature importance (gain)
import lightgbm as lgb

LGB_MODEL_PATH = CKPT / 'lgbm_versionA.txt'
if not LGB_MODEL_PATH.exists():
    print(f'  Skipped LightGBM importance — {LGB_MODEL_PATH} not found.')
else:
    booster_fi = lgb.Booster(model_file=str(LGB_MODEL_PATH))
    gain_imp   = booster_fi.feature_importance(importance_type='gain')
    split_imp  = booster_fi.feature_importance(importance_type='split')
    feat_names = booster_fi.feature_name()

    TOP_N = 20
    top_idx = np.argsort(gain_imp)[-TOP_N:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, imp, title in zip(
        axes,
        [gain_imp[top_idx], split_imp[top_idx]],
        ['Gain (information per split)', 'Split count (usage frequency)']
    ):
        ax.barh(range(TOP_N), imp, color='#2ca02c', alpha=0.8)
        ax.set_yticks(range(TOP_N))
        ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=9)
        ax.set_xlabel(title)
        ax.set_title(f'Top {TOP_N} features — {title}')

    plt.suptitle('D4a — LightGBM Feature Importance', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'D4a_lgbm_importance.png', bbox_inches='tight')
    plt.show()
    print('Saved → D4a_lgbm_importance.png')

    # %% D4b — LightGBM SHAP values (TreeSHAP, fast even on 20M rows)
    try:
        import shap
        print('\n  Computing SHAP values on 5,000 test samples...')
        testX_lgb = colStack(testA, ALL_FEATS_LGB).astype(np.float64)
        shap_sample = testX_lgb[:5_000]

        explainer  = shap.TreeExplainer(booster_fi)
        shap_vals  = explainer.shap_values(shap_sample)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_vals, shap_sample,
            feature_names=ALL_FEATS_LGB,
            max_display=20,
            show=False, plot_type='bar',
        )
        plt.title('D4b — LightGBM SHAP Feature Importance (mean |SHAP|)')
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'D4b_lgbm_shap.png', bbox_inches='tight')
        plt.show()
        print('Saved → D4b_lgbm_shap.png')

        # Beeswarm plot (shows direction of effect)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_vals, shap_sample,
            feature_names=ALL_FEATS_LGB,
            max_display=20, show=False,
        )
        plt.title('D4b — LightGBM SHAP Beeswarm (direction of feature effect)')
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'D4b_lgbm_shap_beeswarm.png', bbox_inches='tight')
        plt.show()
        print('Saved → D4b_lgbm_shap_beeswarm.png')

    except ImportError:
        print('  shap not installed — skipping SHAP. Run: pip install shap')


# %% D4c — LR top coefficients
# Load saved model or retrain on full training set (~15 min if retraining).
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
import joblib

LR_MODEL_PATH = CKPT / 'lr_model.pkl'
ohe_lr, scaler_lr, lr_model = None, None, None

if LR_MODEL_PATH.exists():
    print('\n  Loading saved LR model...')
    lr_model = joblib.load(LR_MODEL_PATH)
    # Reconstruct OHE and scaler from trainA (needed for feature names)
    trainA_lr = pl.read_parquet(WIDE / 'trainA.parquet')
    ohe_lr = OneHotEncoder(sparse_output=True, handle_unknown='ignore', dtype=np.float32)
    ohe_lr.fit(colStack(trainA_lr, OHE_FEATS).astype(np.int32))
    scaler_lr = StandardScaler()
    scaler_lr.fit(colStack(trainA_lr, DENSE_FEATS_LR).astype(np.float32))
    print('  LR model loaded.')
else:
    print('\n  lr_model.pkl not found — retraining LR on full training set (~15 min)...')
    trainA_lr = pl.read_parquet(WIDE / 'trainA.parquet')
    ohe_lr = OneHotEncoder(sparse_output=True, handle_unknown='ignore', dtype=np.float32)
    ohe_lr.fit(colStack(trainA_lr, OHE_FEATS).astype(np.int32))
    scaler_lr = StandardScaler()
    scaler_lr.fit(colStack(trainA_lr, DENSE_FEATS_LR).astype(np.float32))

    def buildLRMatrix(df):
        ohe_p   = ohe_lr.transform(colStack(df, OHE_FEATS).astype(np.int32))
        dense_p = csr_matrix(scaler_lr.transform(
            colStack(df, DENSE_FEATS_LR).astype(np.float32)))
        return hstack([ohe_p, dense_p], format='csr')

    t0 = time.time()
    trainX_lr = buildLRMatrix(trainA_lr)
    trainY_lr  = trainA_lr['clk'].to_numpy()
    lr_model = SGDClassifier(
        loss='log_loss', penalty='l2', alpha=1e-6,
        max_iter=3, tol=1e-4, random_state=42, verbose=0,
    )
    lr_model.fit(trainX_lr, trainY_lr)
    print(f'  Retrained in {(time.time()-t0)/60:.1f} min')
    joblib.dump(lr_model, CKPT / 'lr_model.pkl')
    print(f'  Saved → {CKPT / "lr_model.pkl"}')

if lr_model is not None and ohe_lr is not None:
    # Build feature name list: [ohe categories expanded] + [dense features]
    ohe_feat_names = []
    for i, col in enumerate(OHE_FEATS):
        for cat in ohe_lr.categories_[i]:
            ohe_feat_names.append(f'{col}={int(cat)}')
    all_feat_names_lr = ohe_feat_names + DENSE_FEATS_LR

    coefs = np.abs(lr_model.coef_[0])
    top_lr = np.argsort(coefs)[-25:]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(25), coefs[top_lr], color='#888888', alpha=0.8)
    ax.set_yticks(range(25))
    ax.set_yticklabels([all_feat_names_lr[i] for i in top_lr], fontsize=9)
    ax.set_xlabel('|Coefficient| (absolute value)')
    ax.set_title('D4c — LR Top 25 Feature Coefficients')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'D4c_lr_coefficients.png', bbox_inches='tight')
    plt.show()
    print('Saved → D4c_lr_coefficients.png')


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL — PER-SEGMENT AUC BREAKDOWN
# Required models: ALL available models
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('ADDITIONAL: Per-Segment AUC Breakdown')
print('='*60)

segments = {
    'Overall':               np.ones(len(testY), dtype=bool),
    'User: cold  (<10)':    (testA['user_imp_count'] <  10).to_numpy(),
    'User: warm (10-100)':  (testA['user_imp_count'].is_between(10, 100)).to_numpy(),
    'User: hot   (>100)':   (testA['user_imp_count'] > 100).to_numpy(),
    'Has profile':           (testA['has_profile'] == 1).to_numpy(),
    'No profile':            (testA['has_profile'] == 0).to_numpy(),
    'Low activity  (<100ev)':(testA['user_total_events'] <  100).to_numpy(),
    'High activity (≥500ev)':(testA['user_total_events'] >= 500).to_numpy(),
    'Night (0-6h)':          testA['hour'].is_between(0,  5).to_numpy(),
    'Peak  (20-23h)':        testA['hour'].is_between(20, 23).to_numpy(),
    'Low cate_ctr  (<0.03)': (testA['cate_ctr'] < 0.03).to_numpy(),
    'High cate_ctr (>0.07)': (testA['cate_ctr'] > 0.07).to_numpy(),
}

segAuc = {}
for sname, mask in segments.items():
    if testY[mask].sum() < 10:
        continue
    segAuc[sname] = {m: roc_auc_score(testY[mask], allPreds[m][mask]) for m in MODELS}

segNames  = list(segAuc.keys())
aucMatrix = np.array([[segAuc[s].get(m, np.nan) for m in MODELS] for s in segNames])

fig, ax = plt.subplots(figsize=(max(7, nM * 1.8), max(5, len(segNames) * 0.6)))
vmin = max(0.50, np.nanmin(aucMatrix) - 0.02)
vmax = min(0.80, np.nanmax(aucMatrix) + 0.02)
im = ax.imshow(aucMatrix, aspect='auto', cmap='RdYlGn', vmin=vmin, vmax=vmax)
ax.set_xticks(range(nM)); ax.set_xticklabels(MODELS)
ax.set_yticks(range(len(segNames))); ax.set_yticklabels(segNames, fontsize=10)
ax.set_title('Per-Segment AUC Heatmap', fontsize=13)
for i in range(len(segNames)):
    for j in range(nM):
        v = aucMatrix[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8.5)
plt.colorbar(im, ax=ax, shrink=0.8, label='AUC')
plt.tight_layout()
plt.savefig(FIG_DIR / 'seg_auc_heatmap.png', bbox_inches='tight')
plt.show()
print('Saved → seg_auc_heatmap.png')


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL — SAMPLE COMPLEXITY CURVES
# Required models: LR + LightGBM (retrained at 6 fractions)
#                  Neural models plotted as ★ at 100% if predictions exist
# WARNING: this section takes 30–90 minutes to run.
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('ADDITIONAL: Sample Complexity Curves  (30–90 min)')
print('='*60)

FRACTIONS  = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]

if 'trainA_lr' not in dir():
    print('Loading trainA...')
    trainA_lr = pl.read_parquet(WIDE / 'trainA.parquet')

nTrain = len(trainA_lr)

# Fit OHE + scaler on full train (fixed preprocessor — changes reflect data qty only)
if ohe_lr is None:
    ohe_lr = OneHotEncoder(sparse_output=True, handle_unknown='ignore', dtype=np.float32)
    ohe_lr.fit(colStack(trainA_lr, OHE_FEATS).astype(np.int32))
    scaler_lr = StandardScaler()
    scaler_lr.fit(colStack(trainA_lr, DENSE_FEATS_LR).astype(np.float32))

def buildLRMatrix(df):
    return hstack([
        ohe_lr.transform(colStack(df, OHE_FEATS).astype(np.int32)),
        csr_matrix(scaler_lr.transform(colStack(df, DENSE_FEATS_LR).astype(np.float32))),
    ], format='csr')

# Load valA here (deferred from startup to avoid holding 3 large DFs in RAM at once)
print('Loading valA (scalar columns only)...')
valA = (
    pl.scan_parquet(WIDE / 'valA.parquet')
    .drop('behavior_seq')
    .collect()
)
print(f'valA: {valA.shape}')

valX_lr   = buildLRMatrix(valA)
testX_lr  = buildLRMatrix(testA)
testY_arr = testA['clk'].to_numpy()

valX_lgb  = colStack(valA,  ALL_FEATS_LGB).astype(np.float64)
testX_lgb = colStack(testA, ALL_FEATS_LGB).astype(np.float64)
dval_lgb  = lgb.Dataset(
    valX_lgb, label=valA['clk'].to_numpy().astype(np.int32),
    feature_name=ALL_FEATS_LGB, categorical_feature=CAT_IDX_LGB, free_raw_data=False,
)

complexityAuc = {'LR': [], 'LightGBM': []}
trainSizes    = []

for frac in FRACTIONS:
    nSubset = max(int(nTrain * frac), 1000)
    subset  = trainA_lr.head(nSubset)   # chronological — no leakage
    subY    = subset['clk'].to_numpy()
    trainSizes.append(nSubset)
    print(f'\n  [{frac:.0%}]  {nSubset:,} rows')

    t0 = time.time()
    subX_lr = buildLRMatrix(subset)
    lr_tmp = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-6,
                           max_iter=3, tol=1e-4, random_state=42)
    lr_tmp.fit(subX_lr, subY)
    lr_auc = roc_auc_score(testY_arr, lr_tmp.predict_proba(testX_lr)[:, 1])
    complexityAuc['LR'].append(lr_auc)
    print(f'    LR:       AUC={lr_auc:.4f}  ({time.time()-t0:.0f}s)')

    t0 = time.time()
    dtrain_lgb = lgb.Dataset(
        colStack(subset, ALL_FEATS_LGB).astype(np.float64),
        label=subY.astype(np.int32),
        feature_name=ALL_FEATS_LGB, categorical_feature=CAT_IDX_LGB, free_raw_data=True,
    )
    booster_sc = lgb.train(
        {'objective': 'binary', 'metric': 'auc', 'num_leaves': 255,
         'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
         'bagging_freq': 5, 'min_child_samples': 20, 'verbose': -1, 'seed': 42},
        dtrain_lgb, num_boost_round=1000, valid_sets=[dval_lgb], valid_names=['val'],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    lgb_auc = roc_auc_score(
        testY_arr,
        booster_sc.predict(testX_lgb, num_iteration=booster_sc.best_iteration),
    )
    complexityAuc['LightGBM'].append(lgb_auc)
    print(f'    LightGBM: AUC={lgb_auc:.4f}  ({time.time()-t0:.0f}s)')

fig, ax = plt.subplots(figsize=(9, 5))
for m in ['LR', 'LightGBM']:
    ax.plot(trainSizes, complexityAuc[m], marker='o',
            label=m, color=COLORS[m], linewidth=2, zorder=3)
for m in ['Wide&Deep', 'DeepFM', 'DIN']:
    if m in allPreds:
        ax.scatter([nTrain], [roc_auc_score(testY, allPreds[m])],
                   marker='*', s=250, label=m, color=COLORS[m], zorder=5)

ax.set_xscale('log')
ax.set_xlabel('Training set size (log scale)')
ax.set_ylabel('Test AUC')
ax.set_title('Sample Complexity Curves')
ax.legend(loc='lower right')

ax2 = ax.twiny()
ax2.set_xscale('log'); ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(trainSizes)
ax2.set_xticklabels([f'{int(f*100)}%' for f in FRACTIONS], fontsize=9)
ax2.set_xlabel('Fraction of training data')

plt.tight_layout()
plt.savefig(FIG_DIR / 'sample_complexity.png', bbox_inches='tight')
plt.show()
print('Saved → sample_complexity.png')


# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('All figures saved to figures/analysis/')
print('='*60)
