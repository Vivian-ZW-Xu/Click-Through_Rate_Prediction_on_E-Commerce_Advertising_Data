# Point-in-Time CTR Prediction: Project and Experimental Proposal

**Project status:** data preprocessing completed; behavior identity audit passed; the first Logistic Regression baseline exists but its validation-day calibration anomaly must still be resolved.

**Role of this document:** this is the execution contract for the remainder of the project. New experiments must map to a research question below. If the plan changes, update this document first rather than adding an unplanned experiment.

---

## 1. Project objective

Build and evaluate a leakage-free, point-in-time CTR prediction system on the Taobao advertising dataset. The project will not stop at comparing five model scores. It will separate:

1. **Information gain:** what is added by static/context information, aggregated historical behavior, and historical click feedback?
2. **Model gain:** with the same observable information, what is added by linear, tree-based, interaction-based, and neural architectures?
3. **Sequence gain:** does a user's own behavior sequence improve prediction, and does attention add value beyond simple pooling?
4. **Robustness:** where do gains hold or fail under cold start, probability calibration, and temporal drift?
5. **Engineering cost:** what accuracy is obtained for a model's size and inference cost?

The final claim will therefore be about a controlled experimental system, not simply “DIN achieved the highest AUC.”

---

## 2. Research questions

### RQ1 — Information value

How much predictive value is added by each information layer?

- **F1:** static user, ad, product, placement, price, and time context;
- **F2:** leakage-free aggregated user behavior before the impression;
- **F3:** leakage-free, past-only historical click feedback;
- **F4:** ordered historical behavior used by the sequence model.

### RQ2 — Model-family value

When models receive the same observable fields and temporal cutoffs, how do LR, LightGBM, Wide & Deep, and DeepFM differ in ranking, probability quality, robustness, and efficiency?

This is a **common-information benchmark**, not a claim that every model receives an identical tensor. Model-native representations are part of each model's inductive bias.

### RQ3 — Mechanism value

Which architectural component produces an observed improvement?

- Wide & Deep: deep-only versus wide + deep;
- DeepFM: DNN-only, FM-only, and FM + DNN;
- DIN: context-only, mean pooling, attention, and a mismatched-user negative control.

### RQ4 — Generalization and operational value

Do improvements persist for new users, new ads, different activity levels, and a later date? Are the predicted probabilities sufficiently calibrated for pCTR use?

### RQ5 — Post-training calibration

After a ranking model is frozen, how much can a separate calibration layer improve next-day probability quality without changing the underlying ranking model?

---

## 3. Fixed data protocol

### 3.1 Unit of prediction

One row is one ad impression. The label is `clk`.

### 3.2 Time split

All timestamps and date boundaries use **Asia/Shanghai** time.

| Split | Beijing dates | Purpose |
|---|---|---|
| Train | 2017-05-06 to 2017-05-11 | fitting model parameters and preprocessing |
| Validation | 2017-05-12 | hyperparameter/model selection and diagnosis |
| Test | 2017-05-13 | final reporting |

May 13 has already been inspected during baseline development. It is therefore described honestly as the frozen final reporting set, not as a never-seen competition holdout. From this point onward, May 13 must not drive feature choices, hyperparameters, or model selection.

### 3.3 Sampling

- Main results use the full data without negative downsampling.
- Any development subsample must be deterministic, preserve every training date, and never replace the full-data final run.
- All compared models use the same rows for a given experiment.

### 3.4 Point-in-time rules

1. Every join is performed using raw IDs.
2. Categorical encoding occurs only after all joins and is fitted on training data only.
3. A feature for an impression at time/date \(t\) may use only information available before its defined cutoff.
4. Current-row `clk` and future labels are never used as features.
5. Aggregate behavior windows use only completed prior Beijing dates under the current daily-granularity design.
6. Sequence events must precede the impression timestamp.
7. Identity-level spot checks must verify that stored histories belong to the same `user_raw`.

The daily-window design is an explicit approximation: its feature freshness is lower than an hourly or streaming production system. This limitation will be documented rather than hidden.

---

## 4. Information layers

### F1 — Static and contextual information

- raw user and ad identifiers;
- category, campaign, customer, brand, and placement;
- user profile fields;
- cleaned/log price and missing/sentinel indicators;
- Beijing hour and weekday, subject to the weekday validation described in Phase 0.

### F2 — Aggregated historical behavior

For 1-, 3-, 7-, and 14-day windows before the impression date:

- page views;
- favorites;
- cart additions;
- purchases;
- days since last event;
- days since last purchase;
- history-availability indicators.

Counts may be transformed with `log1p` for models that require a stable numeric scale. The same underlying counts and cutoffs are available to every applicable model.

### F3 — Past-only historical click feedback

Candidate entities:

- user;
- ad;
- category;
- campaign;
- customer/advertiser;
- brand, if coverage supports it.

For prediction date \(d\), statistics may use only labeled impressions with Beijing date `< d`. Each entity receives:

- past impressions;
- past clicks;
- smoothed historical CTR;
- optional confidence/coverage indicator.

A basic smoothed estimate is:

\[
\widehat{CTR}_e = \frac{clicks_e + \alpha \cdot CTR_{global}}{impressions_e + \alpha}
\]

The prior and smoothing strength must be estimated from training history only. Unseen or low-frequency entities back off to an appropriate broader prior. No self-inclusive CTR feature is permitted.

Because each user has little labeled impression history in this short window, `user_past_ctr` is expected to be strongly shrunk toward its prior. Its usefulness must not be judged from AUC alone. For every F3 entity, report:

- fraction with zero prior impressions;
- prior-impression distribution (`p0`, `p25`, `p50`, `p75`, `p90`, `p99`);
- smoothing weight \(w_e = n_e/(n_e+\alpha)\);
- fractions with \(w_e < 0.1\), \(0.1 \le w_e < 0.5\), and \(w_e \ge 0.5\);
- coverage and performance by prior-impression bucket.

Therefore, a weak user-level F3 result may mean insufficient evidence rather than absence of stable user propensity.

### F4 — Historical behavior sequence

DIN receives the user's events before the current impression, represented with fields actually present in `behavior_log`, such as:

- category;
- brand;
- behavior type;
- timestamp/time gap;
- valid missing-value indicators.

Sequence construction must use `user_raw`, enforce `event_time < impression_time`, define a maximum sequence length, and pass an identity/timestamp audit before training.

---

## 5. Model families and their roles

| Model | Role in the study | What it can add |
|---|---|---|
| Logistic Regression | additive baseline and information probe | independent linear contributions |
| LightGBM | nonlinear tabular benchmark | thresholds and conditional interactions |
| Wide & Deep | memorization/generalization benchmark | sparse memorization plus neural generalization |
| DeepFM | interaction-learning benchmark | learned second-order interactions plus DNN |
| DIN | sequence-interest model | candidate-conditioned attention over user history |

No model is declared superior solely because it has a higher raw score while using a richer information set.

---

## 6. Experimental design

### Phase 0 — Stabilize the LR baseline

Purpose: ensure the first measuring instrument is valid before using it for feature conclusions.

Current observation:

- validation AUC: 0.602637;
- validation LogLoss: 0.194338;
- validation actual CTR: 0.049299;
- validation mean prediction: 0.059193;
- test mean prediction is close to test CTR.

Required diagnosis:

1. compare the existing LR with an otherwise identical run excluding weekday;
2. inspect convergence, intercept, learning rate, regularization, and per-epoch metrics if weekday does not explain the bias;
3. preserve every run under a distinct name;
4. freeze the LR training recipe only after validation behavior is understood.

This is a bounded diagnostic phase, not the main research contribution.

### Phase 1 — Build one shared evaluator

Every model must write predictions with at least:

- `impression_id`;
- `clk`;
- predicted probability;
- model/run identifier.

The evaluator must compute the metrics and slices defined in Section 8. Model scripts must not implement incompatible private evaluation logic.

### Phase 2 — Build and audit F3

1. construct daily point-in-time impression/click aggregates;
2. apply smoothing and backoff;
3. join using raw IDs;
4. prove that all source labels precede the prediction date;
5. inspect first-day fallback behavior, coverage, and distributions;
6. rebuild model inputs without changing F1/F2 definitions.

### Phase 3 — Information ablation

Use two representative models:

| Model | F1 | F1+F2 | F1+F2+F3 |
|---|---:|---:|---:|
| LR | run | run | run |
| LightGBM | run | run | run |

This phase answers whether information gains survive both a linear and a nonlinear learner. Running all feature combinations across all five models is optional only after the core study is complete.

### Phase 4 — Common-information model benchmark

Using F1+F2+F3:

| Model | Included in benchmark |
|---|---:|
| LR | yes |
| LightGBM | yes |
| Wide & Deep | yes |
| DeepFM | yes |
| DIN context-only backbone | yes; cross-referenced to Phase 6/Table C |

Fairness means common samples, labels, temporal cutoffs, observable information fields, evaluation, and model-selection rules. It does **not** mean identical encodings or tensors. One-hot encoding, tree splits, embeddings, and learned interactions are model mechanisms.

The conclusion is limited to:

> End-to-end performance of model families under a common information budget.

It is not a causal claim that every score difference comes from one named component.

### Phase 5 — Component ablations

Run only the comparisons needed to explain mechanisms:

- **Wide & Deep:** deep-only vs wide + deep;
- **DeepFM:** DNN-only vs FM-only vs DeepFM;
- **LightGBM:** feature-family ablation plus SHAP/importance as descriptive analysis, not causal proof;
- **LR:** coefficients and feature-family ablation, not individual high-dimensional ID interpretation.

### Phase 6 — Sequence study

DIN is evaluated separately because it receives F4 in addition to the common context.

| Variant | Purpose |
|---|---|
| Context-only DIN backbone | sequence-free reference |
| Mean-pooled correct-user history | value of sequence content without candidate attention |
| Attention over correct-user history | incremental value of DIN attention |
| Matched wrong-user history | negative control for user-specific identity |

Wrong-user controls should be matched approximately on history length/activity so that the model cannot win only from trivial length differences.

Order shuffling is included only if the implemented sequence encoder contains positional/time-order information. Vanilla set-like DIN attention is not expected to change merely from permuting the same history items; claiming otherwise would be an invalid ablation.

### Phase 7 — Post-training probability calibration

Calibration is a separate deployable layer after the base model. It is not merely a diagnostic plot.

For every final model:

1. freeze the feature set, architecture, hyperparameters, checkpoint, and raw prediction function;
2. produce raw predictions on May 12;
3. fit both Platt/sigmoid scaling and isotonic regression using May 12 only;
4. choose between them using five-fold user-grouped out-of-fold LogLoss on May 12, with the choice made before examining calibrated May 13 results;
5. refit the chosen calibrator on all May 12 predictions;
6. apply it once to May 13;
7. report raw and calibrated May 13 LogLoss, COPC, Brier score, ECE, and reliability curves;
8. report AUC before/after as a consistency check. Platt scaling should preserve ranking; isotonic regression can create ties.

This simulates a practical next-day calibration workflow: yesterday's labeled traffic calibrates today's pCTR. A base model and its calibrator are saved as separate artifacts. Test data never chooses the calibration method.

### Phase 8 — Robustness and error analysis

Run the final selected models on:

1. cold-start quadrants;
2. user activity buckets;
3. ad frequency buckets;
4. history-length buckets;
5. placement (`pid`) groups;
6. prediction deciles/calibration curves;
7. validation-day versus test-day performance.

Analysis must distinguish an observed association from a demonstrated cause.

### Phase 9 — Public benchmark anchor, engineering evaluation, and final reporting

The DSIN paper reports the following AUCs on the same Alimama advertising dataset:

| Published model | DSIN paper advertising AUC |
|---|---:|
| Wide & Deep | 0.6326 |
| DIN | 0.6330 |

The paper trains on 2017-05-06 through 2017-05-12 and tests on 2017-05-13. Our development protocol holds May 12 out for validation, so ordinary development results are **not exactly protocol-matched**. After all choices are frozen, run one external-anchor retraining for the selected Wide & Deep and DIN configurations on May 6–12, then evaluate May 13. No tuning is allowed after viewing these anchor results.

Published values are reference anchors, not guaranteed reproduction targets: feature construction, preprocessing, sequence representation, and implementation may still differ. The final report must state these differences beside the comparison.

Source: Feng et al., [*Deep Session Interest Network for Click-Through Rate Prediction*](https://arxiv.org/abs/1905.06482), IJCAI 2019, Table 1.

For each final model, report:

- serialized model size;
- batch inference throughput under one fixed hardware/software setup;
- peak or representative inference memory when measurable;
- feature dependencies and sequence requirements.

Then produce reproducible result tables, figures, README instructions, and resume/interview claims based only on the new pipeline's results.

---

## 7. Hyperparameter protocol

### 7.1 General rules

1. Hyperparameters are selected only on validation data.
2. Test metrics never select features, hyperparameters, epochs, or architectures.
3. Each search space and budget is written down before examining its results.
4. Search failures and unsuccessful runs remain in the experiment log.
5. Early stopping is used where supported.
6. The final selected neural configuration is run with three seeds; report mean and standard deviation.
7. We claim “best under the declared search budget,” not a globally optimal model.

### 7.2 Initial budgets

These are default upper bounds and may be reduced only for a documented compute constraint:

| Family | Development budget | Full-data confirmation |
|---|---|---|
| LR | up to 12 learning-rate/L2/epoch configurations | best 2 candidates |
| LightGBM | up to 20 configurations | best 2 candidates |
| Wide & Deep | up to 12 configurations | best 2 candidates; final best with 3 seeds |
| DeepFM | up to 12 configurations | best 2 candidates; final best with 3 seeds |
| DIN | up to 12 configurations after sequence audit | best 2 candidates; final best with 3 seeds |

A deterministic development subset may be used to screen configurations, but finalists must be retrained on the full training set.

### 7.3 Wall-clock budget and stopping rule

From the current project state, the additional end-to-end compute budget on the M4 Max is capped at **60 machine-hours**:

| Work block | Maximum machine-hours |
|---|---:|
| LR stabilization + shared evaluator | 3 |
| F3 construction/audit + LR/LightGBM information study | 12 |
| Wide & Deep + DeepFM tuning, final runs, and component ablations | 20 |
| Sequence construction/audit + DIN tuning and controls | 20 |
| Calibration, benchmark-anchor retraining, robustness, and engineering measurements | 5 |

The core project excluding DIN is capped at **35 machine-hours**. DIN is a high-value extension with a separate **20-hour cap**, plus at most 5 shared finishing hours.

For each model family:

\[
actual\ trial\ count = \min(config\ cap,\ \lfloor family\ hour\ cap / measured\ pilot\ runtime \rfloor)
\]

If a pilot shows that the written configuration count exceeds its hour cap, reduce the number of configurations; do not silently expand the time budget. Full-data confirmation and required ablations take priority over additional search trials. One-time preprocessing code may be optimized if it exceeds its block, but the project does not add models to compensate for unused hours.

### 7.4 Model-selection rule

CTR has both ranking and probability objectives. Therefore:

- validation **LogLoss** selects the default probability model;
- **AUC/GAUC** are co-primary ranking outcomes;
- if the minimum-LogLoss configuration loses more than 0.002 absolute AUC relative to the best-AUC configuration, both configurations are retained as a Pareto trade-off rather than forcing a false single winner;
- calibration is diagnosed separately and is not inferred from LogLoss alone.

---

## 8. Evaluation protocol

### 8.1 Core metrics

| Metric | Question answered | Status |
|---|---|---|
| LogLoss | How good are the predicted probabilities under a proper scoring rule? | core |
| AUC | How well are clicked impressions ranked over non-clicked impressions? | core |
| GAUC | How well does ranking work within eligible users? | core, with eligible-user count and row coverage |
| PR-AUC | How does ranking behave under low click prevalence? | secondary diagnostic |
| COPC | Is mean predicted CTR aligned with actual CTR? | calibration summary |
| Reliability curve | Where is the model over/under-confident? | calibration diagnostic |

LogLoss is not described as a pure calibration metric. It combines discrimination, confidence, and probability quality and can be dominated by highly confident mistakes. Calibration conclusions require COPC and reliability curves.

### 8.2 Metrics deliberately not emphasized

- Accuracy, precision, recall, and F1 depend on an arbitrary classification threshold and are not primary CTR metrics.
- NDCG is not a headline metric unless a defensible request/slate grouping exists.
- ECE and Brier score may appear in an appendix but are not allowed to crowd the main decision table.

### 8.3 Cold-start quadrants

“Seen” means present in the training split's corresponding vocabulary/history; “new” means absent.

| User | Ad | Segment |
|---|---|---|
| seen | seen | established traffic |
| seen | new | ad cold start |
| new | seen | user cold start |
| new | new | double cold start |

Every segment report includes rows, clicks, CTR, AUC when defined, and LogLoss. Undefined AUC is reported as undefined rather than silently dropped.

### 8.4 Uncertainty

- Neural results: mean ± standard deviation over three final seeds.
- Paired user-cluster bootstrap confidence intervals: only for the final headline comparisons, not every experimental cell.
- Proposed headline comparisons: best tabular vs LR, DeepFM vs its DNN-only ablation, DIN attention vs mean pooling/correct negative control.

---

## 9. Main result tables

### Table A — Information gain

| Model | Features | LogLoss | AUC | GAUC | COPC |
|---|---|---:|---:|---:|---:|
| LR | F1 | | | | |
| LR | F1+F2 | | | | |
| LR | F1+F2+F3 | | | | |
| LightGBM | F1 | | | | |
| LightGBM | F1+F2 | | | | |
| LightGBM | F1+F2+F3 | | | | |

### Table B — Common-information model benchmark

| Model | Information | LogLoss | AUC | GAUC | PR-AUC | COPC | Size | Throughput |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LR | F1+F2+F3 | | | | | | | |
| LightGBM | F1+F2+F3 | | | | | | | |
| Wide & Deep | F1+F2+F3 | | | | | | | |
| DeepFM | F1+F2+F3 | | | | | | | |
| DIN context-only | F1+F2+F3 | | | | | | | |

The context-only DIN backbone also appears as the first row of Table C; it is the bridge between the common-information benchmark and the sequence study.

### Table C — Sequence and negative-control study

| Variant | Information | LogLoss | AUC | GAUC | Meaning |
|---|---|---:|---:|---:|---|
| Context-only | F1+F2+F3 | | | | sequence-free reference |
| Mean pool | + correct F4 | | | | sequence content gain |
| DIN attention | + correct F4 | | | | candidate attention gain |
| Wrong-user control | + matched incorrect F4 | | | | identity negative control |

### Table D — Robustness

Final models are compared across the four cold-start quadrants, activity/frequency buckets, and validation/test dates with sample coverage shown.

### Table E — Post-training calibration

| Model | Version | LogLoss | COPC | Brier | ECE | AUC |
|---|---|---:|---:|---:|---:|---:|
| each final model | raw | | | | | |
| each final model | Platt/isotonic selected on validation OOF | | | | | |

Table B's final reporting must show or cross-reference both raw and calibrated LogLoss/COPC. Raw COPC answers which architecture is naturally calibrated; calibrated COPC answers which deployed model pipeline is best after the common calibration stage.

---

## 10. Reproducibility and artifacts

Every run must save:

- immutable run name;
- code/config version;
- feature set (`F1`, `F1+F2`, `F1+F2+F3`, or sequence variant);
- data split and row counts;
- hyperparameters and seed;
- validation and test predictions where allowed;
- metric summary;
- training time and environment;
- model artifact.

Existing results are never silently overwritten. The experiment registry must make failed and successful runs distinguishable.

Old-project numerical conclusions and attention interpretations are not reused. Only results generated by the new point-in-time pipeline may appear in the README, resume, report, or interview narrative.

---

## 11. Execution order and gates

| Order | Work item | Completion gate |
|---:|---|---|
| 0 | Stabilize LR | validation bias explained or bounded; LR recipe frozen |
| 1 | Shared evaluator | one command evaluates any prediction file consistently |
| 2 | Build F3 | leakage, cutoff, smoothing, coverage, and fallback checks pass |
| 3 | LR/LGBM information ablation | Table A complete |
| 4 | Tune common-information models | declared budgets completed; final configs frozen |
| 5 | Model benchmark | Table B complete |
| 6 | Model-component ablations | architecture claims supported by direct ablations |
| 7 | Sequence pipeline and DIN | identity/time audit passes before training |
| 8 | DIN controls | Table C complete |
| 9 | Post-training calibration | calibration method selected without test; Table E complete |
| 10 | Robustness and uncertainty | Table D, calibration plots, selected CIs complete |
| 11 | Public benchmark anchor | frozen Wide & Deep/DIN retrained on May 6–12; comparison caveats recorded |
| 12 | Engineering and reporting | reproducible README, figures, and final claims complete |

Do not advance past a gate merely because a script ran successfully. The required validation or table must also be complete.

---

## 12. Scope control

### Required core project

- correct point-in-time pipeline;
- shared evaluator;
- F1/F2/F3 information ablation with LR and LightGBM;
- common-information comparison of LR, LightGBM, Wide & Deep, and DeepFM;
- calibration and cold-start analysis;
- post-training calibration as a separately saved module;
- reproducible artifacts.

### High-value extension

- audited DIN sequence pipeline;
- mean-pooling, attention, and wrong-user negative control;
- model size and inference throughput.
- protocol-matched public benchmark anchor for frozen Wide & Deep and DIN configurations.

### Optional only after core completion

- all feature combinations across every neural model;
- large-scale rolling-origin evaluation;
- extensive bootstrap across every slice;
- additional architectures not tied to a new research question.

---

## 13. Final interpretation rules

1. **Information gain** is claimed only from feature-set ablations with the model otherwise fixed.
2. **Architecture gain** is claimed only under a common information budget or from a direct component ablation.
3. **Sequence gain** is separated from attention gain.
4. **Calibration** is not inferred from AUC or LogLoss alone.
5. **Causality** is not claimed from SHAP, attention weights, or error correlations.
6. **Best model** always means best under the declared data protocol, metric, and tuning budget.
7. A statistically small improvement is not treated as practically meaningful without examining robustness and engineering cost.
8. Raw-model quality and calibrated-pipeline quality are both reported; post-hoc calibration is never presented as improved ranking.

---

## 14. Immediate next action

Complete **Phase 0** only:

1. preserve the existing LR run;
2. run an otherwise identical LR without weekday;
3. compare validation/test AUC, LogLoss, mean prediction, COPC, and decile calibration;
4. decide whether weekday handling explains the validation-day bias;
5. if not, inspect optimizer convergence and intercept behavior;
6. freeze the LR recipe and move immediately to the shared evaluator.

No new model family is started before this gate is closed.
