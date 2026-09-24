# Fairness: GH2022DHS, tier B (lightgbm)

Budget = poverty rate, one rule for everyone; out-of-fold predictions, folds 1-4.
Overall exclusion error: 32.2%. Intervals: 95% cluster bootstrap.
**Flag** = even the interval's lower end is above the overall exclusion error.

| Grouping | Group | Pop. share | Poor hh | Poverty rate | Selected | Exclusion (95% CI) | Inclusion | Flag |
|---|---|---|---|---|---|---|---|---|
| residence | rural | 48.4% | 1391 | 35.6% | 38.6% | 25.4% (21%-30%) | 31.2% |  |
| residence | urban | 51.6% | 434 | 11.3% | 8.5% | 52.4% (43%-62%) | 36.5% | **worse** |
| head_sex | female | 33.4% | 605 | 18.4% | 16.7% | 39.3% (33%-45%) | 33.3% | **worse** |
| head_sex | male | 66.6% | 1220 | 25.4% | 26.2% | 29.6% (25%-35%) | 31.8% |  |
| children | has children | 85.7% | 1443 | 24.9% | 25.0% | 32.3% (28%-37%) | 32.8% |  |
| children | no children | 14.3% | 382 | 12.3% | 11.4% | 30.4% (23%-38%) | 24.7% |  |
| elderly_only | elderly only | 1.3% | 88 | 28.4% | 30.0% | 5.8% (1%-12%) | 10.9% |  |
| elderly_only | other | 98.7% | 1737 | 23.0% | 23.0% | 32.6% (28%-37%) | 32.6% |  |
| household_size | 1-2 | 14.6% | 457 | 14.5% | 13.6% | 23.1% (18%-29%) | 18.0% |  |
| household_size | 3-4 | 30.2% | 520 | 16.7% | 16.2% | 39.4% (32%-46%) | 37.3% | **worse** |
| household_size | 5-6 | 28.6% | 457 | 22.5% | 21.9% | 31.9% (26%-38%) | 29.8% |  |
| household_size | 7+ | 26.6% | 391 | 35.5% | 37.4% | 30.6% (24%-39%) | 34.0% |  |
| region | Ahafo | 2.3% | 109 | 27.2% | 30.2% | 41.6% (29%-58%) | 47.4% |  |
| region | Ashanti | 18.4% | 66 | 13.2% | 9.7% | 63.1% (48%-80%) | 49.6% | **worse** |
| region | Bono | 3.6% | 63 | 15.2% | 15.9% | 42.2% (22%-65%) | 44.5% |  |
| region | Bono East | 3.8% | 103 | 33.8% | 40.3% | 17.7% (7%-35%) | 31.1% |  |
| region | Central | 10.6% | 72 | 17.3% | 13.8% | 52.7% (38%-72%) | 40.6% | **worse** |
| region | Eastern | 8.8% | 79 | 17.2% | 12.4% | 44.5% (24%-75%) | 23.3% |  |
| region | Greater Accra | 14.6% | 24 | 3.6% | 2.8% | 65.0% (42%-97%) | 54.9% | **worse** |
| region | North East | 2.4% | 250 | 65.9% | 75.3% | 14.7% (8%-24%) | 25.3% |  |
| region | Northern | 9.4% | 247 | 57.0% | 63.2% | 14.0% (9%-21%) | 22.5% |  |
| region | Oti | 2.8% | 140 | 45.1% | 48.4% | 22.5% (14%-38%) | 27.8% |  |
| region | Savannah | 2.6% | 184 | 59.7% | 65.7% | 15.1% (5%-28%) | 22.9% |  |
| region | Upper East | 4.5% | 127 | 30.4% | 32.5% | 37.9% (26%-53%) | 41.8% |  |
| region | Upper West | 2.9% | 153 | 39.6% | 50.0% | 22.9% (15%-32%) | 39.0% |  |
| region | Volta | 4.9% | 94 | 17.0% | 16.9% | 33.4% (22%-45%) | 33.0% |  |
| region | Western | 5.7% | 42 | 11.2% | 9.6% | 57.2% (36%-81%) | 50.2% | **worse** |
| region | Western North | 2.7% | 72 | 20.2% | 15.1% | 48.5% (31%-67%) | 30.8% |  |
