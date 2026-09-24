# Targeting: GH2022DHS, tier B (lightgbm)

Out-of-fold predictions on folds 1-4; population-weighted; test fold unused.

## Calibration check (cross-fitted)

| Method | Brier | ECE | ROC-AUC |
|---|---|---|---|
| none | 0.1038 | 0.0180 | 0.9011 |
| platt | 0.1038 | 0.0093 | 0.9002 |
| isotonic | 0.1045 | 0.0090 | 0.8961 |

Chosen: **none** (Brier decides; ties within 0.0005 keep 'none').

## Targeting curve

| Budget | Score cut-off | Poor reached | Exclusion | Inclusion | Random: poor reached | Random: inclusion |
|---|---|---|---|---|---|---|
| 5% | 0.856 | 18.9% | 81.1% | 13.2% | 5.0% | 76.9% |
| 10% | 0.747 | 35.9% | 64.1% | 17.3% | 10.0% | 76.9% |
| 15% | 0.629 | 48.9% | 51.1% | 24.9% | 15.0% | 76.9% |
| 20% | 0.493 | 61.8% | 38.2% | 28.8% | 20.0% | 76.9% |
| 25% | 0.372 | 71.3% | 28.7% | 34.3% | 25.0% | 76.9% |
| 30% | 0.280 | 78.8% | 21.2% | 39.5% | 30.0% | 76.9% |
| 40% | 0.155 | 88.3% | 11.7% | 49.1% | 40.0% | 76.9% |
| 50% | 0.074 | 94.5% | 5.5% | 56.4% | 50.0% | 76.9% |
| 60% | 0.040 | 97.7% | 2.3% | 62.4% | 60.0% | 76.9% |
