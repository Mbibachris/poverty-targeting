"""Poverty targeting dashboard (Streamlit).

Run locally:
    streamlit run dashboard/app.py

It talks to the prediction API at API_URL (default: the live Azure service).
"""

import io
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from api_client import ApiClient, ApiError, api_url
from ui_logic import (
    curve_at,
    grouped,
    likelihood_band,
    number_bounds,
    option_label,
    parse_households,
    reason_sentence,
    results_rows,
    template_rows,
    verdict,
)

# Targeting curve for the deployed model (cross-validated; see reports/targeting/).
CURVE_SOURCE = os.environ.get(
    "CURVE_SOURCE",
    "https://raw.githubusercontent.com/Mbibachris/poverty-targeting/main/"
    "reports/targeting/GH2022DHS_tierB.json",
)
BATCH_LIMIT = 1000

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


@st.cache_data(ttl=3600, show_spinner=False)
def targeting_curve() -> list[dict] | None:
    """The trade-off curve, from a local file or a web address; None if unavailable."""
    try:
        if CURVE_SOURCE.startswith("http"):
            return requests.get(CURVE_SOURCE, timeout=20).json()["curve"]
        return json.loads(Path(CURVE_SOURCE).read_text())["curve"]
    except (requests.RequestException, OSError, ValueError, KeyError):
        return None


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
    st.subheader("Budget explorer")
    st.write(
        "A programme can only help part of the population. Move the **budget slider** in "
        "the sidebar to see what the model achieves at each level, compared with choosing "
        "households at random."
    )
    curve = targeting_curve()
    if curve is None:
        st.info("The targeting results could not be loaded right now. Please try again later.")
    else:
        point = curve_at(curve, budget_pct / 100)
        a, b, c = st.columns(3)
        a.metric("Poor people reached", f"{point['coverage_poor']:.0%}")
        b.metric("Of those helped, not poor", f"{point['inclusion_error']:.0%}")
        c.metric(
            "Random selection would reach",
            f"{point['random_coverage_poor']:.0%}",
            help="Choosing the same number of households by lottery.",
        )
        st.write(
            f"With a budget covering **{point['budget']:.0%}** of the population, the model "
            f"reaches about **{point['coverage_poor']:.0%} of poor people**; about "
            f"**{point['inclusion_error']:.0%}** of the people it selects are not poor. "
            f"Missed poor people: about **{1 - point['coverage_poor']:.0%}**."
        )
        chart = pd.DataFrame(
            {
                "Budget (% of population)": [row["budget"] * 100 for row in curve],
                "Model: % of poor reached": [row["coverage_poor"] * 100 for row in curve],
                "Random: % of poor reached": [row["random_coverage_poor"] * 100 for row in curve],
            }
        ).set_index("Budget (% of population)")
        st.line_chart(chart)
        st.caption(
            "Estimated on households the model had not seen during training "
            "(cross-validation, Ghana DHS 2022)."
        )

with many:
    st.subheader("Many households")
    st.write(
        "Upload a spreadsheet with one household per row and one column per question. "
        "Start from the template so the column names are right."
    )
    template = pd.DataFrame(template_rows(info["questions"]))
    st.download_button(
        "Download template (CSV)", template.to_csv(index=False), "households_template.csv"
    )
    upload = st.file_uploader("Spreadsheet of households", type=["csv", "xlsx"])
    if upload is not None:
        if upload.name.endswith(".xlsx"):
            table = pd.read_excel(upload, dtype=object)
        else:
            table = pd.read_csv(upload, dtype=object)
        households, problems = parse_households(table.to_dict("records"), info["questions"])
        if len(households) > BATCH_LIMIT:
            problems.append(f"Please upload at most {BATCH_LIMIT} households at a time.")
        if problems:
            st.error("Please fix these entries and upload again:\n- " + "\n- ".join(problems[:20]))
        else:
            with st.spinner(f"Assessing {len(households)} households..."):
                try:
                    predictions = client().predict_batch(households, budget=budget_pct / 100)
                except ApiError as exc:
                    st.error(str(exc))
                else:
                    results = pd.concat([table, pd.DataFrame(results_rows(predictions))], axis=1)
                    selected = sum(p["selected"] for p in predictions)
                    st.success(
                        f"Assessed {len(predictions)} households: {selected} would be "
                        f"selected at a {budget_pct}% budget."
                    )
                    st.dataframe(results, use_container_width=True)
                    buffer = io.StringIO()
                    results.to_csv(buffer, index=False)
                    st.download_button(
                        "Download results (CSV)", buffer.getvalue(), "households_results.csv"
                    )

with about:
    st.subheader("About the model")
    st.markdown(
        f"""
**What it does.** It estimates how likely a household is to be *multidimensionally poor*
(the global MPI used by UNDP and OPHI: deprived at once in schooling, health and living
conditions) from {len(info["questions"])} quick questions. It was trained on the
**Ghana Demographic and Health Survey 2022**.

**What it is for.** Ranking households when a programme can only help some of them, as one
input alongside community validation and a way to appeal. It must **not** decide
eligibility on its own.

**How well it works** (on households it had never seen):
- With a budget covering about a quarter of the population, it reaches roughly **7 in 10**
  poor people, compared with about 2.5 in 10 by random selection.
- It misses about **3 in 10** poor people when the budget equals the poverty rate.

**Who it serves less well.** Poor households in **urban areas** and **female-headed**
households are missed more often than others. Keep this in mind when reviewing results.

**Limits.** It reflects Ghana in 2022; using it elsewhere or years later needs checking
against new data. Some questions (like electricity or cooking fuel) are also part of the
poverty definition itself, which makes the model look somewhat more accurate than a
purely indirect questionnaire would be.

**Transparency.** Every estimate comes with its main reasons. The full methods, validation
against the official Ghana figure, fairness checks and model card are public in the
[project repository](https://github.com/Mbibachris/poverty-targeting).
"""
    )
