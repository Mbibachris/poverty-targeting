import pytest
import requests
from api_client import ApiClient, ApiError, validation_messages


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    """Plays back a scripted list of responses (or exceptions) and records the calls."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_successful_call_returns_json():
    session = FakeSession(FakeResponse(200, {"status": "ok"}))
    client = ApiClient("https://api.example/", session=session)
    assert client.health() == {"status": "ok"}
    method, url, _ = session.calls[0]
    assert (method, url) == ("GET", "https://api.example/health")


def test_waking_service_is_retried(monkeypatch):
    monkeypatch.setattr("api_client.time.sleep", lambda s: None)
    session = FakeSession(requests.ConnectionError(), FakeResponse(200, {"status": "ok"}))
    assert ApiClient("https://api.example", session=session).health()["status"] == "ok"
    assert len(session.calls) == 2


def test_unreachable_service_gives_a_friendly_message(monkeypatch):
    monkeypatch.setattr("api_client.time.sleep", lambda s: None)
    session = FakeSession(*[requests.Timeout()] * 3)
    with pytest.raises(ApiError, match="try again in a minute"):
        ApiClient("https://api.example", session=session).health()


def test_validation_errors_become_readable_lines():
    detail = [
        {
            "loc": ["body", "household", "hh_size"],
            "msg": "Input should be greater than or equal to 1",
        },
        {"loc": ["body", "household", "region"], "msg": "Input should be 'Ahafo' or 'Ashanti'"},
    ]
    lines = validation_messages(detail)
    assert lines[0] == "household.hh_size: Input should be greater than or equal to 1"
    session = FakeSession(FakeResponse(422, {"detail": detail}))
    with pytest.raises(ApiError, match="household.region"):
        ApiClient("https://api.example", session=session).predict({"hh_size": 0})


def test_predict_sends_household_budget_and_top_k():
    session = FakeSession(FakeResponse(200, {"probability": 0.5}))
    ApiClient("https://api.example", session=session).predict({"hh_size": 3}, budget=0.3)
    _, _, kwargs = session.calls[0]
    assert kwargs["json"] == {"household": {"hh_size": 3}, "budget": 0.3, "top_k": 3}


def test_api_url_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("API_URL", "http://localhost:8000/")
    assert ApiClient().base_url == "http://localhost:8000"
