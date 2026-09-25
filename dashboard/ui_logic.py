"""Plain-language presentation of the API's questions and answers.

Pure functions only (no Streamlit), so every rule about wording and conversion can be
tested directly. The questions themselves always come from the API.
"""

# Readable labels for the codes the model uses. Unlisted codes are tidied automatically.
OPTION_LABELS = {
    "lpg": "LPG (cooking gas)",
    "alcohol_ethanol": "Alcohol / ethanol",
    "coal_lignite": "Coal / lignite",
    "crop_residue": "Crop residue",
    "animal_dung": "Animal dung",
    "straw_shrubs": "Straw / shrubs / grass",
    "processed_biomass": "Processed biomass (pellets)",
    "no_cooking": "No food cooked in the household",
    "natural": "Natural (earth, sand, dung)",
    "rudimentary": "Rudimentary (wood planks, palm, bamboo)",
    "finished": "Finished (cement, tiles, wood floor)",
    "piped_dwelling": "Piped into the dwelling",
    "piped_yard": "Piped into the yard or plot",
    "piped_neighbour": "Piped to a neighbour",
    "public_tap": "Public tap / standpipe",
    "tubewell_borehole": "Tube well / borehole",
    "tanker_cart": "Tanker truck / cart",
    "sachet": "Sachet water",
    "surface_water": "River, lake, pond or stream",
    "flush_sewer": "Flush to piped sewer",
    "flush_septic": "Flush to septic tank",
    "flush_pit": "Flush to pit latrine",
    "flush_elsewhere": "Flush to somewhere else",
    "flush_unknown": "Flush, don't know where",
    "flush_biodigester": "Flush to bio-digester",
    "vip_latrine": "Ventilated improved pit (VIP) latrine",
    "pit_with_slab": "Pit latrine with slab",
    "pit_without_slab": "Pit latrine without slab / open pit",
    "no_facility": "No facility (bush, field)",
    "hanging": "Hanging toilet / latrine",
}

# Where each question appears on the form. Questions not listed go under "Other".
GROUPS = {
    "Location": ["region", "urban"],
    "People in the household": ["hh_size", "n_under5", "n_children_5_17", "n_elderly_65plus"],
    "Household head": ["head_sex", "head_age", "head_years_schooling"],
    "Dwelling": ["sleeping_rooms", "electricity", "cooking_fuel", "floor", "water_source",
                 "sanitation"],
    "Assets and finance": ["has_tv", "has_telephone", "has_refrigerator", "has_motorbike",
                           "has_bank_account", "owns_agri_land", "owns_livestock"],
}  # fmt: skip


def option_label(code: str) -> str:
    """'unprotected_well' -> 'Unprotected well'; known codes get a hand-written label."""
    if code in OPTION_LABELS:
        return OPTION_LABELS[code]
    text = code.replace("_", " ")
    return text[:1].upper() + text[1:]


def grouped(questions: list[dict]) -> dict[str, list[dict]]:
    """Questions arranged into the form's sections, keeping the API's own order inside each."""
    by_name = {q["name"]: q for q in questions}
    placed = {name for names in GROUPS.values() for name in names}
    sections = {
        title: [by_name[n] for n in names if n in by_name] for title, names in GROUPS.items()
    }
    sections["Other"] = [q for q in questions if q["name"] not in placed]
    return {title: qs for title, qs in sections.items() if qs}


def number_bounds(openapi: dict, name: str) -> tuple[int, int]:
    """Allowed range of a numeric answer, read from the API's own published schema."""
    field = openapi["components"]["schemas"]["Household"]["properties"][name]
    for candidate in [field, *field.get("anyOf", [])]:
        if candidate.get("type") == "integer":
            return int(candidate.get("minimum", 0)), int(candidate.get("maximum", 100))
    return 0, 100


def answer_text(value) -> str:
    if value is None:
        return "not known"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return option_label(value) if isinstance(value, str) else str(value)


def reason_sentence(reason: dict) -> str:
    """One reason from the API, as a sentence a non-specialist can read."""
    size = abs(reason["effect"])
    strength = "strongly" if size >= 1 else "moderately" if size >= 0.4 else "slightly"
    return (
        f"**{reason['question']}**: {answer_text(reason['answer'])}. "
        f"This {strength} {reason['direction']} the estimated likelihood of being poor."
    )


def likelihood_band(probability: float) -> str:
    if probability >= 0.7:
        return "Very likely to be poor"
    if probability >= 0.5:
        return "Likely to be poor"
    if probability >= 0.3:
        return "Uncertain"
    if probability >= 0.1:
        return "Unlikely to be poor"
    return "Very unlikely to be poor"


def verdict(result: dict) -> str:
    budget = f"{result['budget']:.0%}"
    if result["selected"]:
        return f"Would be selected by a programme covering {budget} of the population."
    return f"Would not be selected by a programme covering {budget} of the population."
