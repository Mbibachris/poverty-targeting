"""Poverty targeting dashboard (Streamlit).

Run locally:
    streamlit run dashboard/app.py

It talks to the prediction API at API_URL (default: the live Azure service).
"""

import streamlit as st
from api_client import ApiClient, ApiError, api_url

st.set_page_config(page_title="Poverty targeting", page_icon="🏠", layout="wide")


@st.cache_resource
def client() -> ApiClient:
    return ApiClient()


@st.cache_data(ttl=3600, show_spinner=False)
def model_info() -> dict:
    return client().model_info()


st.title("Poverty targeting")
st.write(
    "Estimate how likely a household is to be **multidimensionally poor** (global MPI) "
    "from a short questionnaire, and see who a budget-limited programme would reach."
)

with st.spinner("Connecting to the prediction service (it can take up to a minute to wake up)..."):
    try:
        info = model_info()
    except ApiError as exc:
        st.error(str(exc))
        st.stop()

st.success(
    f"Connected. Model **{info['model']}** (version {info['version']}), "
    f"trained on {info['survey_id']}; it asks {len(info['questions'])} questions."
)
st.caption(f"Prediction service: {api_url()}")
