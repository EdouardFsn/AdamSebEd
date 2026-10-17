| date | change | threshold | F1 (val) | notes |
|------|--------|-----------|----------|-------|
| 9-14 | baseline logistic | 0.5 | 0.223 | under-predicts positives |
| 10-14 | + threshold tuning | 0.2 | 0.422 | peak of sweep |
| 10-16 | finer threshold sweep (0.01 steps) | 0.21 | 0.423 | single split, seed 0 |
| 10-17 | categoricals OFF (dropped invalid cols) | 0.20 | 0.422 | single split, 138 cols |
| 10-17 | + one-hot categoricals (12 features) | 0.21 | 0.427 | single split, cols 138 → 237 |
| 10-17 | baseline logistic, 5-fold CV | 0.2 | 0.424 ± 0.003 | introduction of CV |
| 10-17 | + one-hot categoricals, 5-fold CV | 0.2 | 0.427 ± 0.003 | confirms the single-split gain |
| 10-17 | codebook-checked lists (drop invalid/redundant cols, fix missing codes) | 0.2 | 0.426 ± 0.003 | 5-fold CV, same as above within noise, simpler pipeline |
| 10-17 | + _SMOKER3 one-hot | 0.2 | 0.426 ± 0.004 | no effect (±0.001 per fold) |
| 10-17 | + GENHLTH one-hot | 0.2 | 0.427 ± 0.004 | no effect (±0.001 per fold) |
| 10-17 | − _STATE | 0.2 | 0.426 ± 0.003 | worse on 3/5 folds, never better → kept _STATE |
| 10-17 | final: codebook lists + _SMOKER3, GENHLTH one-hot, _STATE kept | 0.2 | 0.427 ± 0.004 | folds 0.428 / 0.430 / 0.420 / 0.431 / 0.425 |