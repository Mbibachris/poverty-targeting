from streamlit.testing.v1 import AppTest


def test_app_explains_itself_when_the_api_is_unreachable(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens on port 9
    monkeypatch.setattr("api_client.RETRIES", 0)
    app = AppTest.from_file("../dashboard/app.py", default_timeout=60).run()
    assert not app.exception
    assert "could not be reached" in app.error[0].value
