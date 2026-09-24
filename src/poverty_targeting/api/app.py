"""The prediction web service.

The model is loaded once, when the app is created; the request schemas are generated
from it, so the API always accepts exactly what the loaded model understands.

Run locally (reload on code changes):
    uvicorn poverty_targeting.api.app:create_app --factory --reload

The model file is read from the POVERTY_TARGETING_MODEL environment variable, or the
default Ghana model in models/ if it is not set.
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from poverty_targeting import config
from poverty_targeting.api.schemas import (
    BatchPrediction,
    Health,
    ModelInfo,
    Prediction,
    QuestionInfo,
    build_household_model,
    build_request_models,
)
from poverty_targeting.serving import ScoringError, load_bundle, score

MODEL_ENV = "POVERTY_TARGETING_MODEL"
DEFAULT_MODEL = config.MODELS_DIR / "GH2022DHS_tierB_lightgbm.joblib"


def model_path_from_env() -> Path:
    return Path(os.environ.get(MODEL_ENV, DEFAULT_MODEL))


def create_app(model_path: Path | None = None) -> FastAPI:
    """Build the app around one trained model (loaded here, once)."""
    path = Path(model_path or model_path_from_env())
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Train it with `python -m poverty_targeting.final` "
            f"or point {MODEL_ENV} at a saved model."
        )
    bundle = load_bundle(path)
    meta = bundle.meta
    Household = build_household_model(bundle)
    PredictRequest, BatchPredictRequest = build_request_models(Household)
    model_tag = f"{meta['name']}@{meta['version']}"

    app = FastAPI(
        title="Poverty targeting API",
        version=meta["version"],
        description=(
            "Estimates the probability that a household is multidimensionally poor "
            f"(global MPI) from short questionnaire answers. Model: {meta['name']}. "
            "For ranking households within a budgeted programme, with human review; "
            "not for automated eligibility decisions."
        ),
    )

    @app.middleware("http")
    async def add_model_header(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Model"] = model_tag  # every response says which model answered
        return response

    @app.exception_handler(ScoringError)
    async def scoring_error(request: Request, exc: ScoringError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get("/health", response_model=Health, tags=["service"])
    def health():
        """Is the service up, and which model is it serving?"""
        return {"status": "ok", "model": meta["name"], "version": meta["version"]}

    @app.get("/model/info", response_model=ModelInfo, tags=["service"])
    def model_info():
        """The questions the model asks (with valid options) and its default budget."""
        return {
            "model": meta["name"],
            "version": meta["version"],
            "trained_at": meta["trained_at"],
            "survey_id": meta["survey_id"],
            "tier": meta["tier"],
            "default_budget": meta["default_budget"],
            "questions": [
                QuestionInfo(
                    name=q.name,
                    kind=q.kind,
                    question=q.question,
                    options=list(q.options) if q.options else None,
                    required=q.required,
                )
                for q in bundle.questions
            ],
        }

    @app.post("/predict", response_model=Prediction, tags=["prediction"])
    def predict(request: PredictRequest):
        """Score one household: probability, selection at the budget, top reasons."""
        [result] = score(bundle, [request.household.model_dump()], request.budget, request.top_k)
        return result

    @app.post("/predict/batch", response_model=BatchPrediction, tags=["prediction"])
    def predict_batch(request: BatchPredictRequest):
        """Score up to 1,000 households in one request (same budget for all)."""
        answers = [h.model_dump() for h in request.households]
        return {"predictions": score(bundle, answers, request.budget, request.top_k)}

    return app
