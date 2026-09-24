# Fairness: GH2022DHS, tier A (logit)

Budget = poverty rate, one rule for everyone; out-of-fold predictions, folds 1-4.
Overall exclusion error: 43.6%. Intervals: 95% cluster bootstrap.
**Flag** = even the interval's lower end is above the overall exclusion error.

| Grouping | Group | Pop. share | Poor hh | Poverty rate | Selected | Exclusion (95% CI) | Inclusion | Flag |
|---|---|---|---|---|---|---|---|---|
| residence | rural | 48.4% | 1391 | 35.6% | 39.8% | 34.7% (28%-42%) | 41.5% |  |
| residence | urban | 51.6% | 434 | 11.3% | 7.4% | 70.0% (58%-81%) | 54.1% | **worse** |
| head_sex | female | 33.4% | 605 | 18.4% | 15.1% | 60.1% (53%-66%) | 51.4% | **worse** |
| head_sex | male | 66.6% | 1220 | 25.4% | 27.1% | 37.6% (31%-46%) | 41.4% |  |
| children | has children | 85.7% | 1443 | 24.9% | 26.0% | 41.0% (35%-48%) | 43.5% |  |
| children | no children | 14.3% | 382 | 12.3% | 5.7% | 75.3% (69%-82%) | 46.9% | **worse** |
| elderly_only | elderly only | 1.3% | 88 | 28.4% | 17.2% | 68.3% (56%-80%) | 47.5% | **worse** |
| elderly_only | other | 98.7% | 1737 | 23.0% | 23.1% | 43.2% (37%-50%) | 43.6% |  |
| household_size | 1-2 | 14.6% | 457 | 14.5% | 6.2% | 74.4% (67%-81%) | 39.6% | **worse** |
| household_size | 3-4 | 30.2% | 520 | 16.7% | 15.2% | 52.6% (46%-60%) | 47.7% | **worse** |
| household_size | 5-6 | 28.6% | 457 | 22.5% | 25.4% | 38.3% (31%-46%) | 45.3% |  |
| household_size | 7+ | 26.6% | 391 | 35.5% | 38.8% | 35.5% (27%-45%) | 41.0% |  |
| region | Ahafo | 2.3% | 109 | 27.2% | 29.5% | 45.7% (34%-64%) | 50.0% |  |
| region | Ashanti | 18.4% | 66 | 13.2% | 5.3% | 90.7% (81%-99%) | 76.7% | **worse** |
| region | Bono | 3.6% | 63 | 15.2% | 7.7% | 78.0% (65%-92%) | 56.1% | **worse** |
| region | Bono East | 3.8% | 103 | 33.8% | 40.8% | 30.2% (19%-46%) | 42.2% |  |
| region | Central | 10.6% | 72 | 17.3% | 7.4% | 85.2% (73%-98%) | 65.2% | **worse** |
| region | Eastern | 8.8% | 79 | 17.2% | 8.8% | 79.9% (68%-95%) | 60.9% | **worse** |
| region | Greater Accra | 14.6% | 24 | 3.6% | 0.4% | 100.0% (100%-100%) | 100.0% | **worse** |
| region | North East | 2.4% | 250 | 65.9% | 89.9% | 6.6% (2%-12%) | 31.5% |  |
| region | Northern | 9.4% | 247 | 57.0% | 78.0% | 7.7% (4%-14%) | 32.6% |  |
| region | Oti | 2.8% | 140 | 45.1% | 62.0% | 24.4% (15%-35%) | 45.0% |  |
| region | Savannah | 2.6% | 184 | 59.7% | 81.3% | 9.4% (3%-19%) | 33.4% |  |
| region | Upper East | 4.5% | 127 | 30.4% | 37.1% | 44.2% (32%-58%) | 54.3% |  |
| region | Upper West | 2.9% | 153 | 39.6% | 58.5% | 21.1% (14%-29%) | 46.6% |  |
| region | Volta | 4.9% | 94 | 17.0% | 10.5% | 79.1% (65%-92%) | 66.3% | **worse** |
| region | Western | 5.7% | 42 | 11.2% | 4.2% | 85.1% (71%-100%) | 60.4% | **worse** |
| region | Western North | 2.7% | 72 | 20.2% | 18.2% | 78.9% (66%-91%) | 76.5% | **worse** |
