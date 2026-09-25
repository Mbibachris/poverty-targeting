"""Poverty targeting dashboard (Streamlit).

Run locally:
    streamlit run dashboard/app.py

It talks to the prediction API at API_URL (default: the live Azure service).
"""

import streamlit as st
from api_client import ApiClient, ApiError, api_url
from ui_logic import (
    grouped,
    likelihood_band,
    number_bounds,
    option_label,
    reason_sentence,
    verdict,
)

st.set_page_config(page_title="Poverty targeting", page_icon="🏠", layout="wide")


@st.cache_resource
def client() -> ApiClient:
    return ApiClient()


@st.cache_data(ttl=3600, show_spinner=False)
def model_info() -> dict:
    return client().model_info()


@st.cache_data(ttl=3600, show_spinner=False)
def openapi() -> dict:
    return client().openapi()


def question_input(question: dict, schema: dict):
    """One form widget for one question; returns the answer in the API's format."""
    name, label = question["name"], question["question"]
    if question["kind"] == "category":
        return st.selectbox(label, question["options"], format_func=option_label, key=name)
    if question["kind"] == "yes_no":
        return st.radio(label, ["No", "Yes"], horizontal=True, key=name) == "Yes"
    low, high = number_bounds(schema, name)
    if not question["required"]:
        if st.checkbox(f"{label}: not known", key=f"{name}_unknown"):
            return None
    return int(st.number_input(label, min_value=low, max_value=high, value=low, key=name))


def show_result(result: dict) -> None:
    probability = result["probability"]
    left, right = st.columns([1, 2])
    with left:
        st.metric("Estimated likelihood of being poor", f"{probability:.0%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(likelihood_band(probability))
    with right:
        (st.success if result["selected"] else st.info)(verdict(result))
        st.caption(
            f"At this budget, households with a likelihood of at least "
            f"{result['score_cutoff']:.0%} are selected."
        )
    st.markdown("**Main reasons**")
    for reason in result["reasons"]:
        st.markdown(f"- {reason_sentence(reason)}")
    if result["missing_answers"]:
        st.caption(
            "Left blank: " + ", ".join(result["missing_answers"]) + ". "
            "The estimate is less certain without them."
        )
    st.caption(
        "This is an estimate to support, not replace, human judgement. "
        "Decisions about real households need review and a way to appeal."
    )


st.title("Poverty targeting")
st.write(
    "Estimate how likely a household is to be **multidimensionally poor** (global MPI) "
    "from a short questionnaire, and see who a budget-limited programme would reach."
)

with st.spinner("Connecting to the prediction service (it can take up to a minute to wake up)..."):
    try:
        info = model_info()
        schema = openapi()
    except ApiError as exc:
        st.error(str(exc))
        st.stop()

with st.sidebar:
    st.header("Programme budget")
    budget_pct = st.slider(
        "Share of the population the programme can cover (%)",
        min_value=1,
        max_value=99,
        value=round(info["default_budget"] * 100),
        key="budget",
    )
    st.caption(
        f"The default, {round(info['default_budget'] * 100)}%, is the share of people "
        "who are MPI-poor in the training data."
    )
    st.divider()
    st.caption(f"Model {info['model']} (version {info['version']})")
    st.caption(f"Prediction service: {api_url()}")

assess, explore, many, about = st.tabs(
    ["Assess a household", "Budget explorer", "Many households", "About the model"]
)

with assess:
    st.subheader("Assess a household")
    st.write("Answer the questions below, then press **Assess household**.")
    with st.form("household"):
        answers = {}
        for title, questions in grouped(info["questions"]).items():
            st.markdown(f"#### {title}")
            columns = st.columns(2)
            for i, question in enumerate(questions):
                with columns[i % 2]:
                    answers[question["name"]] = question_input(question, schema)
        submitted = st.form_submit_button("Assess household", type="primary")

    if submitted:
        with st.spinner("Assessing..."):
            try:
                result = client().predict(answers, budget=budget_pct / 100)
            except ApiError as exc:
                st.error(str(exc))
            else:
                show_result(result)

with explore:
    st.info("Coming in the next step: explore how the budget changes who is reached.")
with many:
    st.info("Coming soon: upload a spreadsheet of households and download their results.")
with about:
    st.info("Coming soon: what this model does, how well it works, and its limits.")
