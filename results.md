| date | change | threshold | F1 (val) | notes |
|------|--------|-----------|----------|-------|
| 9-14 | baseline logistic | 0.5 | 0.223 | under-predicts positives |
| 10-14 | + threshold tuning | 0.2 | 0.422 | peak of sweep |
| 17-10 | baseline logistic | 0.2 | 0.424 ± 0.003 | introduction of CV with 5-fold |
| 17.10 | baseline logistic | 0.2 | 0.427 ± 0.003 | one-hot incoding with CV 5-fold |