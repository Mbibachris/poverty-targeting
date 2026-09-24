import pytest
from fastapi.testclient import TestClient

from poverty_targeting.api.app import MODEL_ENV, create_app
from poverty_targeting.serving import load_bundle, score


@pytest.fixture(scope="module")
def client(bundle_path):
    return TestClient(create_app(bundle_path))


def test_health_names_the_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "TEST_tierB_lightgbm",
        "version": "0.0.0-test",
    }
    assert response.headers["X-Model"] == "TEST_tierB_lightgbm@0.0.0-test"


def test_model_info_lists_the_questions(client):
    info = client.get("/model/info").json()
    questions = {q["name"]: q for q in info["questions"]}
    assert questions["region"]["options"] == ["coast", "north", "south"]
    assert questions["head_age"]["required"] is False
    assert info["default_budget"] == 0.25


def test_predict_returns_a_scored_household(client, household):
    response = client.post("/predict", json={"household": household, "budget": 0.3})
    assert response.status_code == 200
    result = response.json()
    assert 0 < result["probability"] < 1
    assert result["budget"] == 0.3
    assert len(result["reasons"]) == 3


def test_api_matches_the_serving_core(client, bundle_path, household):
    via_api = client.post("/predict", json={"household": household}).json()
    [direct] = score(load_bundle(bundle_path), [household])
    assert via_api["probability"] == direct["probability"]
    assert via_api["selected"] == direct["selected"]


@pytest.mark.parametrize(
    "change",
    [
        {"region": "atlantis"},  # not a trained option
        {"hh_size": 0},  # out of range
        {"favourite_colour": "blue"},  # unknown field
        {"dependent_share": 0.4},  # computed by the server, never sent
        {"hh_size": 2, "n_under5": 3},  # counts exceed household size
    ],
)
def test_bad_households_are_rejected_with_422(client, household, change):
    response = client.post("/predict", json={"household": {**household, **change}})
    assert response.status_code == 422


def test_missing_required_answer_and_bad_budget_are_rejected(client, household):
    del household["hh_size"]
    assert client.post("/predict", json={"household": household}).status_code == 422
    body = {"household": {**household, "hh_size": 7}, "budget": 1.5}
    assert client.post("/predict", json=body).status_code == 422


def test_batch_scores_every_household(client, household):
    response = client.post("/predict/batch", json={"households": [household, household]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
    assert client.post("/predict/batch", json={"households": []}).status_code == 422


def test_model_path_can_come_from_the_environment(monkeypatch, bundle_path):
    monkeypatch.setenv(MODEL_ENV, str(bundle_path))
    assert TestClient(create_app()).get("/health").status_code == 200


def test_missing_model_file_explains_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match=MODEL_ENV):
        create_app(tmp_path / "nothing_here.joblib")
