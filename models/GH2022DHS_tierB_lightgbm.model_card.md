# Model card: GH2022DHS_tierB_lightgbm

## What it does
Estimates the probability that a household is multidimensionally poor under the global_mpi definition, from 24 questionnaire answers. Model: lightgbm (tier B); trained on GH2022DHS (folds 1-4, 7,139 households); package version 0.1.0.

## Intended use
Ranking households for a social-protection programme with a fixed budget, as one input to a process that includes community validation and appeals. **Not** for automated eligibility decisions without human review.

## Test-fold performance (fold 0, used once)
- Households: 1,788; population poverty rate 21.7%
- ROC-AUC 0.892; PR-AUC 0.704; Brier 0.1044; ECE 0.020
- At a 23.1% budget: 69.1% of poor people reached; exclusion error 30.9%; inclusion error 35.4%

## Targeting curve (test fold)

| Budget | Poor reached | Inclusion error |
|---|---|---|
| 5% | 19.9% | 14.4% |
| 10% | 36.2% | 22.4% |
| 15% | 50.4% | 27.3% |
| 20% | 61.4% | 33.7% |
| 25% | 71.7% | 38.3% |
| 30% | 76.9% | 44.4% |
| 40% | 89.0% | 51.8% |
| 50% | 92.0% | 60.2% |
| 60% | 98.2% | 64.6% |

## Who it serves less well (test fold, 95% cluster-bootstrap intervals)
- residence = urban: exclusion 54.1% (39%-69%)
- head_sex = female: exclusion 49.2% (38%-61%)
- region = Greater Accra: exclusion 88.1% (78%-100%)
- region = Upper West: exclusion 47.7% (31%-69%)

## Known limitations
- The label is the global MPI computed from DHS data; child mortality uses births in the last five years (KR), which undercounts; adolescent nutrition is excluded.
- Tier B includes items that are also MPI inputs, so part of its accuracy is the label's own ingredients (see reports/audit and the Tier A benchmark in reports/cv).
- Poor urban and female-headed households are reached less often (see above).
- One country and one survey year; accuracy elsewhere or later is unknown.

## Questions the model uses
- `region`: Region of residence
- `urban`: Is the household in an urban area?
- `hh_size`: How many people live in this household?
- `head_sex`: Sex of the household head
- `head_age`: Age of the household head
- `n_under5`: How many household members are under 5?
- `n_children_5_17`: How many household members are aged 5 to 17?
- `n_elderly_65plus`: How many household members are 65 or older?
- `dependent_share`: (computed) share of members under 18 or aged 65+
- `sleeping_rooms`: How many rooms are used for sleeping?
- `persons_per_room`: (computed) household size per sleeping room
- `has_bank_account`: Does any member have a bank account?
- `owns_agri_land`: Does the household own land usable for agriculture?
- `owns_livestock`: Does the household own livestock or farm animals?
- `head_years_schooling`: Years of schooling completed by the household head
- `electricity`: Does the household have electricity?
- `cooking_fuel`: Main fuel used for cooking
- `floor`: Main floor material
- `water_source`: Main source of drinking water
- `sanitation`: Type of toilet facility
- `has_tv`: Does the household own a television?
- `has_telephone`: Does the household own a mobile or landline telephone?
- `has_refrigerator`: Does the household own a refrigerator?
- `has_motorbike`: Does the household own a motorcycle or scooter?
