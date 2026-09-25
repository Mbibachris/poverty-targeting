import pytest
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from poverty_targeting.api.app import create_app

APP = "../dashboard/app.py"


@pytest.fixture(autouse=True)
def fresh_caches():
    """Each test starts with an empty Streamlit cache, so no client leaks between tests."""
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def test_app_explains_itself_when_the_api_is_unreachable(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens on port 9
    monkeypatch.setattr("api_client.RETRIES", 0)
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert "could not be reached" in app.error[0].value


def test_household_assessment_end_to_end(monkeypatch, bundle_path):
    """The dashboard, talking to a real in-memory API that serves the small test model."""
    api = TestClient(create_app(bundle_path))
    monkeypatch.setenv("API_URL", "http://testserver")
    monkeypatch.setattr("api_client.requests.Session", lambda: api)

    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert app.selectbox(key="region").options == ["Coast", "North", "South"]
    assert app.checkbox(key="head_age_unknown")  # optional answer can be left blank

    app.selectbox(key="region").select("north")
    app.number_input(key="hh_size").set_value(7)
    app.button[0].click().run()  # "Assess household"

    assert not app.exception
    assert app.metric[0].label == "Estimated likelihood of being poor"
    assert app.metric[0].value.endswith("%")
    assert any("Would" in box.value for box in [*app.success, *app.info])
    reasons = [m.value for m in app.markdown if "estimated likelihood of being poor" in m.value]
    assert len(reasons) == 3
