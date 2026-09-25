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


# --- budget explorer ------------------------------------------------------------------


def curve_at(curve: list[dict], budget: float) -> dict:
    """Targeting outcomes at any budget, interpolated between the measured points."""
    points = sorted(curve, key=lambda row: row["budget"])
    budget = min(max(budget, points[0]["budget"]), points[-1]["budget"])
    for low, high in zip(points, points[1:], strict=False):
        if low["budget"] <= budget <= high["budget"]:
            share = (budget - low["budget"]) / (high["budget"] - low["budget"])
            return {
                key: low[key] + share * (high[key] - low[key])
                for key in ("coverage_poor", "inclusion_error", "random_coverage_poor")
            } | {"budget": budget}
    return {**points[-1], "budget": budget}


# --- many households ------------------------------------------------------------------

YES = {"yes", "y", "true", "1"}
NO = {"no", "n", "false", "0"}


def template_rows(questions: list[dict]) -> list[dict]:
    """One example row for the spreadsheet template, using each question's first option."""
    example = {}
    for q in questions:
        if q["kind"] == "category":
            example[q["name"]] = q["options"][0]
        elif q["kind"] == "yes_no":
            example[q["name"]] = "no"
        else:
            example[q["name"]] = "" if not q["required"] else 1
    return [example]


def _blank(value) -> bool:
    return (
        value is None or (isinstance(value, float) and value != value) or str(value).strip() == ""
    )


def parse_households(rows: list[dict], questions: list[dict]) -> tuple[list[dict], list[str]]:
    """Spreadsheet rows -> API households, plus readable problems (row numbers as in Excel)."""
    households, problems = [], []
    for i, row in enumerate(rows, start=2):  # row 1 is the header in a spreadsheet
        household = {}
        for q in questions:
            value = row.get(q["name"])
            if _blank(value):
                if q["required"]:
                    problems.append(f"Row {i}: '{q['name']}' is empty")
                household[q["name"]] = None
            elif q["kind"] == "yes_no":
                text = str(value).strip().lower()
                if text in YES or text in NO:
                    household[q["name"]] = text in YES
                else:
                    problems.append(f"Row {i}: '{q['name']}' should be yes or no, got '{value}'")
            elif q["kind"] == "number":
                try:
                    household[q["name"]] = int(float(value))
                except ValueError:
                    problems.append(f"Row {i}: '{q['name']}' should be a number, got '{value}'")
            else:
                household[q["name"]] = str(value).strip()
        households.append({k: v for k, v in household.items() if v is not None})
    return households, problems


def results_rows(predictions: list[dict]) -> list[dict]:
    """Flat, spreadsheet-friendly results: one row per household, same order as uploaded."""
    return [
        {
            "likelihood_poor": p["probability"],
            "selected": "yes" if p["selected"] else "no",
            "main_reason": p["reasons"][0]["question"] if p["reasons"] else "",
            "main_reason_direction": p["reasons"][0]["direction"] if p["reasons"] else "",
        }
        for p in predictions
    ]
