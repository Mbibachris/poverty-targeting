# Model comparison: GH2022DHS

Selection rule: keep 'logit' unless a challenger wins in >= 3 of 4 folds and gains >= 0.005 mean ROC-AUC. Folds 1-4 only; test fold unused.

| Tier | Model | ROC-AUC | PR-AUC | ECE | Exclusion @ budget | AUC gain vs logit (folds won) | Chosen |
|---|---|---|---|---|---|---|---|
| A | logit | 0.826 | 0.602 | 0.015 | 43.6% | - | yes |
| A | lightgbm | 0.825 | 0.587 | 0.020 | 42.9% | -0.0014 (2/4) |  |
| B | logit | 0.896 | 0.725 | 0.015 | 32.7% | - |  |
| B | lightgbm | 0.901 | 0.732 | 0.018 | 32.2% | +0.0056 (3/4) | yes |
