"""Request and response schemas, generated from the loaded model.

The Household schema is built at start-up from the model's own questions: categories
become fixed choices (the exact options seen in training), yes/no answers booleans,
counts bounded integers. Unknown or computed fields are rejected. FastAPI uses these
schemas to validate every request before it reaches the model (bad input -> HTTP 422)
and to generate the interactive documentation at /docs.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from poverty_targeting.serving import ModelBundle

# Plausible ranges for numeric answers (inclusive). Outside these, it is a typo.
NUMBER_BOUNDS = {
    "hh_size": (1, 60),
    "head_age": (10, 110),
    "n_under5": (0, 30),
    "n_children_5_17": (0, 40),
    "n_elderly_65plus": (0, 20),
    "sleeping_rooms": (0, 40),
    "head_years_schooling": (0, 30),
}
DEFAULT_BOUNDS = (0, 100)
AGE_COUNTS = ("n_under5", "n_children_5_17", "n_elderly_65plus")


class HouseholdBase(BaseModel):
    """Shared behaviour: no unknown fields, and member counts must fit the household."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def counts_fit_household_size(self):
        values = self.model_dump()
        if "hh_size" in values and all(c in values for c in AGE_COUNTS):
            counted = sum(values[c] for c in AGE_COUNTS)
            if counted > values["hh_size"]:
                raise ValueError(
                    f"members counted by age ({counted}) exceed household size "
                    f"({values['hh_size']})"
                )
        return self


def build_household_model(bundle: ModelBundle) -> type[BaseModel]:
    """A Pydantic model with one field per question the loaded model asks."""
    fields = {}
    for q in bundle.questions:
        if q.kind == "category":
            annotation = Literal[q.options]  # only options seen in training
            extra = {}
        elif q.kind == "yes_no":
            annotation, extra = bool, {}
        else:
            low, high = NUMBER_BOUNDS.get(q.name, DEFAULT_BOUNDS)
            annotation, extra = int, {"ge": low, "le": high}
        if q.required:
            fields[q.name] = (annotation, Field(..., description=q.question, **extra))
        else:
            fields[q.name] = (
                annotation | None,
                Field(None, description=f"{q.question} (may be left blank)", **extra),
            )
    return create_model("Household", __base__=HouseholdBase, **fields)


class Reason(BaseModel):
    feature: str
    question: str
    answer: str | float | bool | None
    effect: float = Field(description="Contribution to the log-odds of being poor")
    direction: Literal["raises", "lowers"]


class Prediction(BaseModel):
    probability: float = Field(description="Estimated probability of being MPI-poor")
    budget: float = Field(description="Share of the population the programme can cover")
    score_cutoff: float = Field(description="Lowest score selected at this budget")
    selected: bool = Field(description="Would be selected at this budget")
    reasons: list[Reason]
    missing_answers: list[str]


class Options(BaseModel):
    """Settings shared by single and batch requests."""

    model_config = ConfigDict(extra="forbid")

    budget: float | None = Field(
        None, ge=0.01, le=0.99, description="Share of population covered (default: poverty rate)"
    )
    top_k: int = Field(3, ge=1, le=10, description="How many reasons to return")


def build_request_models(household: type[BaseModel]) -> tuple[type[BaseModel], type[BaseModel]]:
    """(single request, batch request) schemas around a Household schema."""
    single = create_model("PredictRequest", __base__=Options, household=(household, ...))
    batch = create_model(
        "BatchPredictRequest",
        __base__=Options,
        households=(list[household], Field(..., min_length=1, max_length=1000)),
    )
    return single, batch


class BatchPrediction(BaseModel):
    predictions: list[Prediction]


class Health(BaseModel):
    status: Literal["ok"]
    model: str
    version: str


class QuestionInfo(BaseModel):
    name: str
    kind: Literal["category", "yes_no", "number"]
    question: str
    options: list[str] | None
    required: bool


class ModelInfo(BaseModel):
    model: str
    version: str
    trained_at: str
    survey_id: str
    tier: str
    default_budget: float
    questions: list[QuestionInfo]
