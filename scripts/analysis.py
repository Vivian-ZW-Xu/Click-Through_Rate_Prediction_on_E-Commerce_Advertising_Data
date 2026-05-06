"""Cross-model analysis for the Alimama CTR project.

Analysis 1 — Per-segment AUC breakdown:
    Slice the test set by user activity, profile, time of day, and category
    signal; compute each model's AUC on every slice.

Analysis 2 — Model agreement:
    For each actual click, how many models successfully ranked it highly?
    Characterise "universally easy" vs "universally hard" clicks by feature.

Analysis 3 — Sample complexity curves:
    Retrain LR and LightGBM on increasing fractions of the training set.
    Neural models (DIN, DeepFM, Wide&Deep) are plotted as single star points
    at 100% — retraining them multiple times is too expensive.

Prerequisites (run the five train_*.py scripts first):
    checkpoints/preds_lr.npy
    checkpoints/preds_lgbm.npy
    checkpoints/preds_wide_deep.npy
    checkpoints/preds_deepfm.npy
    checkpoints/preds_din.npy
    checkpoints/test_labels.npy

Partial runs are fine — missing model files are skipped gracefully.

Run: `python scripts/analysis.py`
"""

# %% Setup
import polars as pl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

DATA_DIR = Path('data/processed') if Path('data/processed').exists() else Path('../data/processed')
WIDE     = DATA_DIR / 'wide'
CKPT     = Path('checkpoints') if Path('checkpoints').exists() else Path('../checkpoints')
FIG_DIR  = Path('figures/analysis')
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ['LR', 'LightGBM', 'Wide&Deep', 'DeepFM', 'DIN']
PRED_FILES  = ['preds_lr', 'preds_lgbm', 'preds_wide_deep', 'preds_deepfm', 'preds_din']
COLORS = {
    'LR':        '#888888',
    'LightGBM':  '#2ca02c',
    'Wide&Deep': '#ff7f0e',
    'DeepFM':    '#1f77b4',
    'DIN':       '#d62728',
}


# %% Load predictions (skip missing files)
print('Loading predictions...')
testY = np.load(CKPT / 'test_labels.npy')
print(f'Test set: {len(testY):,} samples  CTR={testY.mean():.4f}')

allPreds = {}
for name, fname in zip(MODEL_NAMES, PRED_FILES):
    path = CKPT / f'{fname}.npy'
    if path.exists():
        allPreds[name] = np.load(path)
        auc = roc_auc_score(testY, allPreds[name])
        print(f'  {name:12s}: AUC={auc:.4f}')
    else:
        print(f'  {name:12s}: not found — skipped')

MODELS = list(allPreds.keys())
print(f'\n{len(MODELS)} model(s) loaded: {MODELS}')
if not MODELS:
    raise SystemExit('No predictions found. Run training scripts first.')


# %% Load test features for segmentation
print('\nLoading testA features...')
testA = pl.read_parquet(WIDE / 'testA.parquet')
print(f'testA: {testA.shape}')


# =============================================================================
# ANALYSIS 1 — Per-segment AUC breakdown
# =============================================================================

# %% Define segments
segments = {
    'Overall':              np.ones(len(testY), dtype=bool),
    # --- User activity (impression history in train) ---
    'User: cold  (<10)':   (testA['user_imp_count'] <  10).to_numpy(),
    'User: warm (10-100)': (testA['user_imp_count'].is_between(10, 100)).to_numpy(),
    'User: hot   (>100)':  (testA['user_imp_count'] > 100).to_numpy(),
    # --- Profile availability ---
    'Has profile':         (testA['has_profile'] == 1).to_numpy(),
    'No profile':          (testA['has_profile'] == 0).to_numpy(),
    # --- Behavior richness (events from behavior log) ---
    'Low activity  (<100)': (testA['user_total_events'] <  100).to_numpy(),
    'High activity (≥500)': (testA['user_total_events'] >= 500).to_numpy(),
    # --- Time of day (Beijing hour) ---
    'Night (0–6h)':        testA['hour'].is_between(0,  5).to_numpy(),
    'Peak  (20–23h)':      testA['hour'].is_between(20, 23).to_numpy(),
    # --- Category CTR signal ---
    'Low cate_ctr  (<0.03)': (testA['cate_ctr'] < 0.03).to_numpy(),
    'High cate_ctr (>0.07)': (testA['cate_ctr'] > 0.07).to_numpy(),
}

print('\nSegment sizes:')
for sname, mask in segments.items():
    n, n_pos = mask.sum(), testY[mask].sum()
    print(f'  {sname:27s}: {n:>8,} samples  CTR={100*n_pos/n:.2f}%')


# %% Compute (segment × model) AUC matrix
print('\nComputing per-segment AUC...')
segAuc = {}
for sname, mask in segments.items():
    if testY[mask].sum() < 10:
        continue
    segAuc[sname] = {}
    for model, preds in allPreds.items():
        segAuc[sname][model] = roc_auc_score(testY[mask], preds[mask])

segNames  = list(segAuc.keys())
aucMatrix = np.array([[segAuc[s].get(m, np.nan) for m in MODELS] for s in segNames])

# Print table
print(f'\n{"Segment":<27}', end='')
for m in MODELS:
    print(f'  {m:>10}', end='')
print()
print('-' * (27 + 13 * len(MODELS)))
for sname in segNames:
    print(f'{sname:<27}', end='')
    for m in MODELS:
        v = segAuc[sname].get(m, np.nan)
        print(f'  {v:>10.4f}', end='')
    print()


# %% Figure 1a — heatmap
fig, ax = plt.subplots(figsize=(max(7, len(MODELS) * 1.8), max(5, len(segNames) * 0.6)))
vmin = max(0.50, np.nanmin(aucMatrix) - 0.02)
vmax = min(0.80, np.nanmax(aucMatrix) + 0.02)
im = ax.imshow(aucMatrix, aspect='auto', cmap='RdYlGn', vmin=vmin, vmax=vmax)

ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels(MODELS, fontsize=11)
ax.set_yticks(range(len(segNames)))
ax.set_yticklabels(segNames, fontsize=10)
ax.set_title('Test AUC by Model and Segment', fontsize=13, pad=10)

for i in range(len(segNames)):
    for j in range(len(MODELS)):
        v = aucMatrix[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8.5)

plt.colorbar(im, ax=ax, label='AUC', shrink=0.8)
plt.tight_layout()
plt.savefig(FIG_DIR / 'seg_auc_heatmap.png', bbox_inches='tight')
plt.show()
print('Saved → figures/analysis/seg_auc_heatmap.png')


# %% Figure 1b — bar chart: user activity breakdown
activitySegs = ['User: cold  (<10)', 'User: warm (10-100)', 'User: hot   (>100)']
x     = np.arange(len(activitySegs))
width = 0.75 / len(MODELS)

fig, ax = plt.subplots(figsize=(9, 5))
for i, model in enumerate(MODELS):
    vals = [segAuc[s].get(model, np.nan) for s in activitySegs]
    offset = i * width - (len(MODELS) - 1) * width / 2
    ax.bar(x + offset, vals, width, label=model, color=COLORS.get(model))

ax.set_xticks(x)
ax.set_xticklabels(['Cold\n(imp<10)', 'Warm\n(10≤imp≤100)', 'Hot\n(imp>100)'])
ax.set_ylabel('Test AUC')
ax.set_title('Model AUC by User Activity Level')
ax.legend(loc='lower right')
ylo = max(0.50, np.nanmin(aucMatrix) - 0.03)
yhi = min(0.85, np.nanmax(aucMatrix) + 0.03)
ax.set_ylim(ylo, yhi)
plt.tight_layout()
plt.savefig(FIG_DIR / 'seg_auc_user_activity.png', bbox_inches='tight')
plt.show()
print('Saved → figures/analysis/seg_auc_user_activity.png')


# =============================================================================
# ANALYSIS 2 — Model agreement
# =============================================================================

# %% Binarize each model at top-10% threshold (≈ 2× global CTR of ~5%)
print('\n--- Model Agreement Analysis ---')
TOPK = 0.10

binaryPreds = {}
for model, preds in allPreds.items():
    thresh = np.quantile(preds, 1 - TOPK)
    binaryPreds[model] = (preds >= thresh).astype(np.int8)

# element-wise sum: how many models predicted positive for each sample
agreementScore = sum(binaryPreds.values())
nModels = len(MODELS)
totalClicks = int(testY.sum())

print(f'\nTop-{TOPK*100:.0f}% threshold; {nModels} models')
print(f'\n{"Agreement":>12}  {"N samples":>10}  {"Actual clicks":>14}  {"Recall%":>9}  {"CTR (precision)":>16}')
print('-' * 70)
for n in range(nModels + 1):
    mask = agreementScore == n
    n_samples   = int(mask.sum())
    clicks_here = int(testY[mask].sum())
    recall_pct  = 100 * clicks_here / totalClicks
    ctr_here    = testY[mask].mean() if n_samples > 0 else 0
    print(f'  {n} model(s):  {n_samples:>10,}  {clicks_here:>14,}  {recall_pct:>8.1f}%  {ctr_here:>15.4f}')


# %% Figure 2a — agreement distribution + CTR per level
levels = list(range(nModels + 1))
n_clicks = [int(testY[agreementScore == n].sum())                  for n in levels]
n_nonclk = [int((testY[agreementScore == n] == 0).sum())           for n in levels]
ctr_lvl  = [testY[agreementScore == n].mean() if (agreementScore == n).any() else 0
             for n in levels]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.bar(levels, n_nonclk, label='Non-click', color='#aec7e8')
ax.bar(levels, n_clicks,  label='Click',    color='#d62728', bottom=n_nonclk)
ax.set_xlabel(f'Number of models (out of {nModels}) predicting positive')
ax.set_ylabel('Sample count (log scale)')
ax.set_yscale('log')
ax.set_title('Sample distribution by agreement level')
ax.legend()

ax = axes[1]
ax.bar(levels, ctr_lvl, color='#ff7f0e')
ax.axhline(testY.mean(), color='gray', linestyle='--',
           label=f'Global CTR ({testY.mean():.3f})')
ax.set_xlabel(f'Number of models (out of {nModels}) predicting positive')
ax.set_ylabel('CTR (precision)')
ax.set_title('Precision at each agreement level')
ax.legend()

plt.suptitle('Model Agreement Analysis', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'model_agreement.png', bbox_inches='tight')
plt.show()
print('Saved → figures/analysis/model_agreement.png')


# %% Characterise universally-easy vs universally-hard CLICKS
easyMask = (testY == 1) & (agreementScore == nModels)
hardMask = (testY == 1) & (agreementScore == 0)
allClkMask = testY == 1

print(f'\nEasy clicks (caught by ALL {nModels} models): {easyMask.sum():,}')
print(f'Hard clicks (caught by NO model):              {hardMask.sum():,}')

compareFeatures = [
    'user_imp_count', 'user_ctr', 'ad_imp_count', 'ad_ctr',
    'cate_ctr', 'user_cate_ctr', 'user_total_events', 'user_buy_rate',
]

print(f'\n{"Feature":<22}  {"Easy clicks":>12}  {"Hard clicks":>12}  {"All clicks":>12}')
print('-' * 65)
for col in compareFeatures:
    arr = testA[col].to_numpy().astype(float)
    easy_m = arr[easyMask].mean()  if easyMask.sum() > 0 else float('nan')
    hard_m = arr[hardMask].mean()  if hardMask.sum() > 0 else float('nan')
    all_m  = arr[allClkMask].mean()
    print(f'{col:<22}  {easy_m:>12.4f}  {hard_m:>12.4f}  {all_m:>12.4f}')


# %% Figure 2b — feature distributions for easy vs hard clicks
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(compareFeatures):
    ax   = axes[i]
    arr  = testA[col].to_numpy().astype(float)
    cap  = np.percentile(arr[allClkMask], 98)
    bins = np.linspace(0, cap, 30)

    if easyMask.sum() > 0:
        ax.hist(np.clip(arr[easyMask], 0, cap), bins=bins, alpha=0.6,
                density=True, color='#2ca02c', label=f'Easy (all {nModels} catch)')
    if hardMask.sum() > 0:
        ax.hist(np.clip(arr[hardMask], 0, cap),  bins=bins, alpha=0.6,
                density=True, color='#d62728', label='Hard (none catch)')

    ax.set_title(col, fontsize=10)
    ax.set_xlabel('')
    ax.set_yticks([])
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle('Feature Distributions: Easy vs Hard Clicks (among actual clicks)', fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / 'easy_vs_hard_clicks.png', bbox_inches='tight')
plt.show()
print('Saved → figures/analysis/easy_vs_hard_clicks.png')


# =============================================================================
# ANALYSIS 3 — Sample complexity curves
# =============================================================================

print('\n--- Sample Complexity Curves ---')
print('Retraining LR and LightGBM on data fractions (this may take ~30-90 min total).')
print('Neural models appear as star markers at 100% using saved predictions.\n')

import time
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
import lightgbm as lgb

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]

# --- Feature definitions (mirrors train_lr.py and train_lgbm.py) ---
OHE_FEATS = [
    'pid', 'cate_id', 'cms_segid', 'cms_group_id', 'final_gender_code',
    'age_level', 'pvalue_level', 'shopping_level', 'occupation',
    'new_user_class_level', 'has_profile', 'hour', 'day_of_week',
]
DENSE_FEATS_LR = [
    'price', 'log_price',
    'user_imp_count', 'user_ctr', 'ad_imp_count', 'ad_ctr',
    'cate_ctr', 'user_cate_ctr',
    'user_total_events', 'user_total_pv', 'user_total_buy',
    'user_total_cart', 'user_total_fav', 'user_n_unique_cates', 'user_buy_rate',
]
CAT_FEATS_LGB  = OHE_FEATS  # same low/mid-cardinality set
HCARD_FEATS    = ['user', 'adgroup_id', 'campaign_id', 'customer']
DENSE_FEATS_LGB = DENSE_FEATS_LR
ALL_FEATS_LGB  = CAT_FEATS_LGB + HCARD_FEATS + DENSE_FEATS_LGB
CAT_IDX_LGB    = list(range(len(CAT_FEATS_LGB)))


def colStack(df, cols):
    return np.column_stack([df[c].to_numpy() for c in cols])


# %% Load train / val (needed for complexity retraining)
print('Loading trainA and valA...')
trainA = pl.read_parquet(WIDE / 'trainA.parquet')
valA   = pl.read_parquet(WIDE / 'valA.parquet')
nTrain = len(trainA)
print(f'trainA: {nTrain:,} rows  |  valA: {len(valA):,} rows')

# Fit OHE and scaler on the FULL training set — preprocessor is fixed across
# all fractions so that changes in AUC reflect only data quantity, not encoding.
print('\nFitting OHE + scaler on full train...')
ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore', dtype=np.float32)
ohe.fit(colStack(trainA, OHE_FEATS).astype(np.int32))
scaler = StandardScaler()
scaler.fit(colStack(trainA, DENSE_FEATS_LR).astype(np.float32))


def buildLRMatrix(df):
    ohePart   = ohe.transform(colStack(df, OHE_FEATS).astype(np.int32))
    densePart = csr_matrix(
        scaler.transform(colStack(df, DENSE_FEATS_LR).astype(np.float32))
    )
    return hstack([ohePart, densePart], format='csr')


print('Prebuilding val/test matrices...')
valX_lr   = buildLRMatrix(valA)
testX_lr  = buildLRMatrix(testA)
testY_arr = testA['clk'].to_numpy()

valX_lgb  = colStack(valA,  ALL_FEATS_LGB).astype(np.float64)
testX_lgb = colStack(testA, ALL_FEATS_LGB).astype(np.float64)

dval_lgb = lgb.Dataset(
    valX_lgb, label=valA['clk'].to_numpy().astype(np.int32),
    feature_name=ALL_FEATS_LGB, categorical_feature=CAT_IDX_LGB,
    free_raw_data=False,
)

# %% Training loop across fractions
complexityAuc = {'LR': [], 'LightGBM': []}
trainSizes    = []

for frac in FRACTIONS:
    nSubset = max(int(nTrain * frac), 1000)
    # Take FIRST nSubset rows — chronological order preserves the time-based
    # split and prevents any future data from leaking into smaller subsets.
    subset = trainA.head(nSubset)
    subY   = subset['clk'].to_numpy()
    trainSizes.append(nSubset)
    print(f'\n  [{frac:.0%}]  {nSubset:,} training rows')

    # LR
    t0 = time.time()
    subX_lr = buildLRMatrix(subset)
    lr_model = SGDClassifier(
        loss='log_loss', penalty='l2', alpha=1e-6,
        max_iter=3, tol=1e-4, random_state=42,
    )
    lr_model.fit(subX_lr, subY)
    lr_auc = roc_auc_score(testY_arr, lr_model.predict_proba(testX_lr)[:, 1])
    complexityAuc['LR'].append(lr_auc)
    print(f'    LR:       AUC={lr_auc:.4f}  ({time.time()-t0:.0f}s)')

    # LightGBM
    t0 = time.time()
    subX_lgb = colStack(subset, ALL_FEATS_LGB).astype(np.float64)
    dtrain_lgb = lgb.Dataset(
        subX_lgb, label=subY.astype(np.int32),
        feature_name=ALL_FEATS_LGB, categorical_feature=CAT_IDX_LGB,
        free_raw_data=True,
    )
    lgb_params = {
        'objective': 'binary', 'metric': 'auc',
        'num_leaves': 255, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_child_samples': 20, 'cat_smooth': 10, 'verbose': -1, 'seed': 42,
    }
    booster = lgb.train(
        lgb_params, dtrain_lgb, num_boost_round=1000,
        valid_sets=[dval_lgb], valid_names=['val'],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    lgb_auc = roc_auc_score(
        testY_arr,
        booster.predict(testX_lgb, num_iteration=booster.best_iteration),
    )
    complexityAuc['LightGBM'].append(lgb_auc)
    print(f'    LightGBM: AUC={lgb_auc:.4f}  ({time.time()-t0:.0f}s)  '
          f'best_iter={booster.best_iteration}')


# %% Figure 3 — sample complexity curves
fig, ax = plt.subplots(figsize=(9, 5))

# LR and LightGBM: full learning curves
for model in ['LR', 'LightGBM']:
    ax.plot(trainSizes, complexityAuc[model],
            marker='o', label=model, color=COLORS[model], linewidth=2, zorder=3)

# Neural models: single star at 100% from saved predictions
neuralModels = ['Wide&Deep', 'DeepFM', 'DIN']
for model in neuralModels:
    if model in allPreds:
        auc_full = roc_auc_score(testY, allPreds[model])
        ax.scatter([nTrain], [auc_full], marker='*', s=250,
                   label=model, color=COLORS[model], zorder=5)

ax.set_xscale('log')
ax.set_xlabel('Training set size (log scale)')
ax.set_ylabel('Test AUC')
ax.set_title('Sample Complexity Curves')
ax.legend(loc='lower right')

# Secondary x-axis showing fraction labels
ax2 = ax.twiny()
ax2.set_xscale('log')
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(trainSizes)
ax2.set_xticklabels([f'{int(f*100)}%' for f in FRACTIONS], fontsize=9)
ax2.set_xlabel('Fraction of training data')

plt.tight_layout()
plt.savefig(FIG_DIR / 'sample_complexity.png', bbox_inches='tight')
plt.show()
print('Saved → figures/analysis/sample_complexity.png')


# =============================================================================
# Summary
# =============================================================================
print('\n' + '=' * 60)
print('Figures saved to figures/analysis/:')
print('  seg_auc_heatmap.png        (Analysis 1)')
print('  seg_auc_user_activity.png  (Analysis 1)')
print('  model_agreement.png        (Analysis 2)')
print('  easy_vs_hard_clicks.png    (Analysis 2)')
print('  sample_complexity.png      (Analysis 3)')
print('=' * 60)
