"""Score two example households with the trained model (run after `final`).

python examples/score_household.py
"""

import json

from poverty_targeting import config
from poverty_targeting.serving import load_bundle, score

RURAL_NORTH = {
    "region": "Northern", "urban": False, "hh_size": 6, "head_sex": "female",
    "head_age": 45, "n_under5": 2, "n_children_5_17": 2, "n_elderly_65plus": 0,
    "sleeping_rooms": 1, "has_bank_account": False, "owns_agri_land": True,
    "owns_livestock": True, "head_years_schooling": 0, "electricity": False,
    "cooking_fuel": "wood", "floor": "natural", "water_source": "unprotected_well",
    "sanitation": "no_facility", "has_tv": False, "has_telephone": True,
    "has_refrigerator": False, "has_motorbike": False,
}  # fmt: skip

URBAN_ACCRA = {
    **RURAL_NORTH, "region": "Greater Accra", "urban": True, "n_under5": 0,
    "sleeping_rooms": 3, "has_bank_account": True, "head_years_schooling": 16,
    "electricity": True, "cooking_fuel": "lpg", "floor": "finished",
    "water_source": "piped_dwelling", "sanitation": "flush_sewer", "has_tv": True,
    "has_refrigerator": True,
}  # fmt: skip

if __name__ == "__main__":
    bundle = load_bundle(config.MODELS_DIR / "GH2022DHS_tierB_lightgbm.joblib")
    print(json.dumps(score(bundle, [RURAL_NORTH, URBAN_ACCRA], budget=0.25), indent=2))
