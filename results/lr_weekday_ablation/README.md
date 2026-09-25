# LR Weekday Ablation

The validation date, 2017-05-12, is Friday. Friday is absent from the
2017-05-06 through 2017-05-11 training split.

Removing the untrained discrete weekday feature changed validation results:

- AUC: 0.602637 to 0.602658
- LogLoss: 0.194338 to 0.193406
- mean prediction: 0.059193 to 0.049825
- actual CTR: 0.049299
- COPC: 0.832862 to 0.989462
- prediction bias: 20.07% to 1.07%

Ranking remained effectively unchanged while validation calibration improved
substantially. Test performance was not harmed.

Decision: exclude discrete `day_of_week_idx` from the common development
feature set. Remaining within-decile calibration error is handled in the
post-training calibration phase.
