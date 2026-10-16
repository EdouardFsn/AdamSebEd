| date | change | threshold | F1 (val) | notes |
|------|--------|-----------|----------|-------|
| 9-14 | baseline logistic | 0.5 | 0.223 | under-predicts positives |
| 10-14 | + threshold tuning | 0.2 | 0.422 | peak of sweep |
| 10-16 | + one-hot categoricals (13 cols -> 106 indicators) | 0.2 | 0.428 | 3 splits: 0.421 -> 0.427 mean, +0.005 on each |