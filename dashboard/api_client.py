"""Talking to the prediction API: the dashboard's only connection to the model.

The dashboard never loads the model itself. Every answer comes from the API, so the
dashboard can never disagree with it, and updating the model needs no dashboard change.
"""

import os
import time

import requests

DEFAULT_API_URL = (
    "https://poverty-targeting-api.mangograss-dd837579.spaincentral.azurecontainerapps.io"
)
# The API scales to zero when idle; waking it can take up to about a minute.
TIMEOUT_SECONDS = 90
RETRIES = 2


class ApiError(Exception):
    """A problem talking to the API, with a message fit to show a user."""


def api_url() -> str:
    """The API address: the API_URL environment variable, or the live Azure service."""
    return os.environ.get("API_URL", DEFAULT_API_URL).rstrip("/")


def validation_messages(detail) -> list[str]:
    """Turn the API's 422 error details into readable lines."""
    if isinstance(detail, str):
        return [detail]
    lines = []
    for item in detail:
        field = ".".join(str(part) for part in item.get("loc", []) if part != "body")
        lines.append(f"{field}: {item.get('msg', 'invalid')}" if field else item.get("msg", ""))
    return lines


class ApiClient:
    def __init__(self, base_url: str | None = None, session=None):
        self.base_url = (base_url or api_url()).rstrip("/")
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}{path}"
        for attempt in range(RETRIES + 1):
            try:
                response = self.session.request(method, url, timeout=TIMEOUT_SECONDS, **kwargs)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == RETRIES:
                    raise ApiError(
                        "The prediction service could not be reached. It may be starting "
                        "up after a quiet period; please try again in a minute."
                    ) from exc
                time.sleep(2**attempt)
                continue
            if response.status_code == 422:
                lines = validation_messages(response.json().get("detail", "invalid input"))
                raise ApiError("Some answers need correcting:\n- " + "\n- ".join(lines))
            if response.status_code >= 400:
                raise ApiError(
                    f"The prediction service returned an error ({response.status_code})."
                )
            return response.json()
        raise ApiError("The prediction service could not be reached.")  # pragma: no cover

    def health(self) -> dict:
        return self._request("GET", "/health")

    def model_info(self) -> dict:
        return self._request("GET", "/model/info")

    def predict(self, household: dict, budget: float | None = None, top_k: int = 3) -> dict:
        body = {"household": household, "budget": budget, "top_k": top_k}
        return self._request("POST", "/predict", json=body)

    def predict_batch(self, households: list[dict], budget: float | None = None) -> list[dict]:
        body = {"households": households, "budget": budget}
        return self._request("POST", "/predict/batch", json=body)["predictions"]
