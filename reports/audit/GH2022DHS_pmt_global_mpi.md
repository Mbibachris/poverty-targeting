# Circularity audit: GH2022DHS, feature set 'pmt' vs Global MPI (OPHI/UNDP, 2018 revision)

Somers' D of each feature alone (training folds only; test fold untouched).
0 = no association, +/-1 = perfect. Categorical features show strength only.

| Feature | Tier | Shares label inputs with | D with MPI-poor | Strongest indicator |
|---|---|---|---|---|
| region | A | - | +0.44 | school_attendance (+0.55) |
| urban | A | - | -0.33 | electricity (-0.43) |
| hh_size | A | - | +0.21 | school_attendance (+0.64) |
| head_sex | A | - | +0.02 | assets (+0.24) |
| head_age | A | - | +0.07 | cooking_fuel (+0.28) |
| n_under5 | A | - | +0.25 | nutrition (+0.50) |
| n_children_5_17 | A | - | +0.17 | school_attendance (+0.65) |
| n_elderly_65plus | A | - | +0.04 | cooking_fuel (+0.12) |
| dependent_share | A | - | +0.27 | school_attendance (+0.44) |
| sleeping_rooms | A | - | +0.07 | school_attendance (+0.34) |
| persons_per_room | A | - | +0.17 | school_attendance (+0.36) |
| has_bank_account | A | - | -0.35 | cooking_fuel (-0.38) |
| owns_agri_land | A | - | +0.19 | cooking_fuel (+0.30) |
| owns_livestock | A | - | +0.23 | cooking_fuel (+0.38) |
| head_years_schooling | B | years_schooling | -0.60 | years_schooling (-0.79) |
| electricity | B | electricity | -0.39 | electricity (-1.00) |
| cooking_fuel | B | cooking_fuel | +0.49 | cooking_fuel (+1.00) |
| floor | B | housing | +0.17 | housing (+0.40) |
| water_source | B | drinking_water | +0.57 | drinking_water (+0.79) |
| sanitation | B | sanitation | +0.53 | sanitation (+0.74) |
| has_tv | B | assets | -0.46 | assets (-0.68) |
| has_telephone | B | assets | -0.13 | assets (-0.31) |
| has_refrigerator | B | assets | -0.38 | cooking_fuel (-0.44) |
| has_motorbike | B | assets | +0.02 | assets (-0.31) |

## Full matrix

| Feature | nutrition | child_mortality | years_schooling | school_attendance | cooking_fuel | sanitation | drinking_water | electricity | housing | assets | mpi_poor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| region | +0.28 | +0.30 | +0.37 | +0.55 | +0.43 | +0.23 | +0.45 | +0.33 | +0.43 | +0.17 | +0.44 |
| urban | -0.10 | -0.11 | -0.26 | -0.30 | -0.36 | -0.22 | -0.37 | -0.43 | -0.28 | -0.22 | -0.33 |
| hh_size | +0.41 | +0.38 | -0.04 | +0.64 | +0.41 | -0.04 | +0.25 | +0.13 | +0.14 | -0.17 | +0.21 |
| head_sex | +0.11 | +0.13 | +0.06 | +0.13 | +0.05 | +0.01 | +0.11 | +0.07 | +0.10 | +0.24 | +0.02 |
| head_age | -0.11 | -0.13 | +0.14 | +0.11 | +0.28 | -0.13 | +0.04 | +0.09 | +0.02 | +0.09 | +0.07 |
| n_under5 | +0.50 | +0.30 | +0.06 | +0.36 | +0.22 | +0.07 | +0.16 | +0.12 | +0.12 | -0.05 | +0.25 |
| n_children_5_17 | +0.18 | +0.26 | +0.04 | +0.65 | +0.33 | -0.01 | +0.20 | +0.09 | +0.11 | -0.07 | +0.17 |
| n_elderly_65plus | -0.05 | -0.05 | +0.08 | +0.00 | +0.12 | -0.03 | +0.05 | +0.02 | -0.00 | +0.06 | +0.04 |
| dependent_share | +0.18 | +0.13 | +0.24 | +0.44 | +0.37 | +0.04 | +0.18 | +0.12 | +0.09 | +0.13 | +0.27 |
| sleeping_rooms | +0.15 | +0.11 | -0.05 | +0.34 | +0.20 | -0.26 | +0.18 | +0.03 | +0.08 | -0.24 | +0.07 |
| persons_per_room | +0.32 | +0.35 | -0.00 | +0.36 | +0.31 | +0.17 | +0.10 | +0.11 | +0.08 | +0.02 | +0.17 |
| has_bank_account | -0.10 | -0.11 | -0.37 | -0.28 | -0.38 | -0.32 | -0.25 | -0.30 | -0.22 | -0.32 | -0.35 |
| owns_agri_land | +0.12 | +0.08 | +0.12 | +0.20 | +0.30 | +0.10 | +0.22 | +0.14 | +0.17 | -0.02 | +0.19 |
| owns_livestock | +0.16 | +0.13 | +0.13 | +0.30 | +0.38 | +0.09 | +0.30 | +0.27 | +0.20 | -0.05 | +0.23 |
| head_years_schooling | -0.14 | -0.17 | -0.79 | -0.51 | -0.56 | -0.38 | -0.35 | -0.43 | -0.28 | -0.33 | -0.60 |
| electricity | -0.11 | -0.13 | -0.27 | -0.33 | -0.21 | -0.16 | -0.32 | -1.00 | -0.28 | -0.29 | -0.39 |
| cooking_fuel | +0.24 | +0.19 | +0.37 | +0.45 | +1.00 | +0.41 | +0.47 | +0.46 | +0.39 | +0.28 | +0.49 |
| floor | +0.03 | +0.08 | +0.11 | +0.11 | +0.09 | +0.06 | +0.13 | +0.17 | +0.40 | +0.09 | +0.17 |
| water_source | +0.26 | +0.25 | +0.42 | +0.50 | +0.63 | +0.42 | +0.79 | +0.58 | +0.46 | +0.29 | +0.57 |
| sanitation | +0.24 | +0.16 | +0.43 | +0.48 | +0.57 | +0.74 | +0.48 | +0.50 | +0.36 | +0.25 | +0.53 |
| has_tv | -0.09 | -0.07 | -0.38 | -0.34 | -0.32 | -0.28 | -0.33 | -0.65 | -0.30 | -0.68 | -0.46 |
| has_telephone | -0.01 | +0.01 | -0.14 | -0.05 | -0.05 | -0.05 | -0.04 | -0.09 | -0.05 | -0.31 | -0.13 |
| has_refrigerator | -0.15 | -0.13 | -0.31 | -0.29 | -0.44 | -0.42 | -0.30 | -0.40 | -0.26 | -0.41 | -0.38 |
| has_motorbike | +0.10 | +0.14 | -0.02 | +0.12 | +0.05 | -0.01 | +0.06 | -0.04 | +0.02 | -0.31 | +0.02 |
