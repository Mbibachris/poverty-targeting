# Targeting: GH2022DHS, tier A (logit)

Out-of-fold predictions on folds 1-4; population-weighted; test fold unused.

## Calibration check (cross-fitted)

| Method | Brier | ECE | ROC-AUC |
|---|---|---|---|
| none | 0.1307 | 0.0151 | 0.8257 |
| platt | 0.1310 | 0.0115 | 0.8241 |
| isotonic | 0.1319 | 0.0101 | 0.8196 |

Chosen: **none** (Brier decides; ties within 0.0005 keep 'none').

## Targeting curve

| Budget | Score cut-off | Poor reached | Exclusion | Inclusion | Random: poor reached | Random: inclusion |
|---|---|---|---|---|---|---|
| 5% | 0.722 | 17.3% | 82.7% | 20.4% | 5.0% | 76.9% |
| 10% | 0.604 | 31.7% | 68.3% | 26.8% | 10.0% | 76.9% |
| 15% | 0.506 | 42.6% | 57.4% | 34.6% | 15.0% | 76.9% |
| 20% | 0.420 | 51.5% | 48.5% | 40.6% | 20.0% | 76.9% |
| 25% | 0.354 | 59.9% | 40.1% | 44.8% | 25.0% | 76.9% |
| 30% | 0.297 | 66.2% | 33.8% | 49.1% | 30.0% | 76.9% |
| 40% | 0.207 | 77.6% | 22.4% | 55.2% | 40.0% | 76.9% |
| 50% | 0.140 | 85.6% | 14.4% | 60.5% | 50.0% | 76.9% |
| 60% | 0.097 | 91.4% | 8.6% | 64.9% | 60.0% | 76.9% |
