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
| 10-18 | dependent features OFF (reference, same config as final) | 0.2 | 0.427 ± 0.004 | 5-fold CV, cols 243, folds 0.428 / 0.430 / 0.420 / 0.431 / 0.425 |
| 10-18 | + dependent features (skip-logic fill, DEPENDENT_RULES) | 0.2 | 0.426 ± 0.004 | 5-fold CV, cols 244, folds 0.428 / 0.429 / 0.420 / 0.430 / 0.424, no effect (≤0.001 per fold): info already in precursor columns |
| 10-19 | auto-threshold CV (cat+dep) | chosen t≈0.21 (range 0.20–0.22) | 0.426 ± 0.005 | no notes |
| 10-19 | λ sweep (tune.py): λ=0, γ=0.1, 500 it (previous run.py) | 0.2 | 0.426 ± 0.004 | 5-fold CV, train loss 0.2235: GD not converged |
| 10-19 | λ=0, γ=0.5, 500 it | 0.2 | 0.428 ± 0.003 | train loss 0.2208 (converged), +0.0014, better on 4/5 folds |
| 10-19 | λ=1e-4, γ=0.5 | 0.2 | 0.428 ± 0.003 | same as λ=0 |
| 10-19 | λ=1e-3, γ=0.5 | 0.2 | 0.428 ± 0.003 | same as λ=0 (+0.0003, noise) |
| 10-19 | λ=1e-2, γ=0.5 | 0.2 | 0.415 ± 0.002 | worse; still 0.424 at its best t=0.24 |
| 10-19 | λ=1e-1, γ=0.5 | 0.2 | 0.201 ± 0.001 | t=0.2 no longer fits (probabilities pushed up); 0.414 at its best t=0.39 |
| 10-19 | chosen: λ=0, γ=0.5, 500 it (run.py) | 0.2 | 0.428 ± 0.003 | L2 regularization does not help: no overfitting with 262k rows / 244 cols |
| 10-19 | chosen config (λ=0, γ=0.5) with auto-threshold CV (run.py) | chosen t 0.21 / 0.21 / 0.21 / 0.20 / 0.20 | 0.428 ± 0.003 | vs 0.426 ± 0.005 with γ=0.1 (row above sweep): gain holds under honest threshold selection |
