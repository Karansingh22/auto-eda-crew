"""Streamlit UI for the CrewAI dataset analysis assistant."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from app_logic import (
    MAX_DATASET_BYTES,
    SUPPORTED_DATASET_EXTENSIONS,
    build_dataset_profile,
    clear_generated_reports,
    dataset_profile_to_context,
    dataset_profile_to_prompt,
    dataset_profile_to_route_context,
    load_dataset_from_bytes,
    report_paths,
    validate_dataset_upload,
)
from crew_runner import kickoff_with_retry


TARGET_PLACEHOLDER = "Select target variable"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(file_name: str, file_bytes: bytes):
    """Cache uploaded dataset parsing between Streamlit reruns."""

    return load_dataset_from_bytes(file_name, file_bytes)


st.set_page_config(page_title="CrewAI Dataset Analyst", page_icon=":bar_chart:", layout="wide")

st.title("CrewAI Dataset Analyst")
st.caption("Agent-led chat for greetings and questions. Full supervised EDA + ML analysis for uploaded datasets.")

st.info(
    "Upload a CSV or Excel dataset up to 30 MB, choose the target variable, then ask for "
    "EDA, ML recommendations, data quality checks, or a real-world problem-solving plan."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_reports" not in st.session_state:
    st.session_state.show_reports = False

dataset_profile = None
selected_target = None
uploaded_file = None
dataset_error = None

with st.sidebar:
    st.header("Settings")
    st.info("Model: `groq/llama-3.3-70b-versatile`")

    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.show_reports = False
        clear_generated_reports(OUTPUT_DIR)
        st.rerun()

    st.divider()
    st.header("Dataset")
    max_mb = MAX_DATASET_BYTES // (1024 * 1024)
    supported = ", ".join(f".{ext}" for ext in SUPPORTED_DATASET_EXTENSIONS)
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=list(SUPPORTED_DATASET_EXTENSIONS),
        help=f"Maximum size: {max_mb} MB. Supported formats: {supported}.",
    )

    if uploaded_file is not None:
        try:
            size_bytes = int(getattr(uploaded_file, "size", 0))
            validate_dataset_upload(uploaded_file.name, size_bytes)
            file_bytes = uploaded_file.getvalue()
            df = load_uploaded_dataset(uploaded_file.name, file_bytes)

            st.success(f"Loaded `{uploaded_file.name}`")
            st.write(f"Rows: `{df.shape[0]}`")
            st.write(f"Columns: `{df.shape[1]}`")

            selected_target = st.selectbox(
                "Target variable",
                [TARGET_PLACEHOLDER, *[str(column) for column in df.columns]],
                index=0,
            )

            if selected_target != TARGET_PLACEHOLDER:
                dataset_profile = build_dataset_profile(
                    df,
                    file_name=uploaded_file.name,
                    size_bytes=len(file_bytes),
                    target_column=selected_target,
                )
                st.success(f"Target selected: `{selected_target}`")
            else:
                st.warning("Select the target variable before running EDA or ML analysis.")

            with st.expander("Preview dataset", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)

        except Exception as exc:  # Streamlit should show user-friendly upload errors.
            dataset_error = str(exc)
            st.error(dataset_error)

st.subheader("Real-World Workflow")
st.markdown(
    """
This project can solve practical tabular-data problems such as churn prediction, loan risk,
sales forecasting, lead scoring, fraud triage, pricing analysis, operations bottlenecks,
and customer segmentation. The key is to upload the dataset, select the target, and ask a
business-focused question like: "Find the biggest drivers of churn and recommend an ML plan."
""".strip()
)

if dataset_profile is not None:
    with st.expander("Current dataset profile", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", dataset_profile.row_count)
        col2.metric("Columns", dataset_profile.column_count)
        col3.metric("Target", dataset_profile.target_column)
        st.write("Target summary")
        st.json(dataset_profile.target_summary)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Say hi, ask a question, or request dataset EDA/ML analysis..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if uploaded_file is not None and dataset_profile is None:
            response = (
                "Please select a valid target variable first. After that I can run EDA, "
                "data quality checks, and ML recommendations for your dataset."
            )
            st.session_state.show_reports = False
            st.markdown(response)

        else:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            if dataset_profile is not None:
                with st.spinner("Letting the orchestrator choose chat or analysis..."):
                    from agent.agents import analysis_crew, chat_crew, route_crew

                    clear_generated_reports(OUTPUT_DIR)
                    route_context = dataset_profile_to_route_context(prompt, dataset_profile)
                    route = str(
                        kickoff_with_retry(route_crew, inputs={"topic": route_context})
                    ).strip().upper()

                if "ANALYSIS" in route:
                    with st.spinner("Running EDA and supervised ML agents on your dataset profile..."):
                        topic = dataset_profile_to_prompt(prompt, dataset_profile)
                        result = kickoff_with_retry(analysis_crew, inputs={"topic": topic})
                        response = str(result)
                    st.session_state.show_reports = True
                else:
                    with st.spinner("Answering..."):
                        context = dataset_profile_to_context(prompt, dataset_profile)
                        result = kickoff_with_retry(chat_crew, inputs={"topic": context})
                        response = str(result)
                    st.session_state.show_reports = False

            else:
                with st.spinner("Answering..."):
                    from agent.agents import chat_crew

                    clear_generated_reports(OUTPUT_DIR)
                    result = kickoff_with_retry(chat_crew, inputs={"topic": prompt})
                    response = str(result)
                st.session_state.show_reports = False

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

if st.session_state.show_reports:
    eda_report, ml_report = report_paths(OUTPUT_DIR)
    col1, col2 = st.columns(2)

    if eda_report.exists():
        with col1:
            with st.expander("EDA Report", expanded=False):
                st.markdown(eda_report.read_text(encoding="utf-8"))

    if ml_report.exists():
        with col2:
            with st.expander("ML Report", expanded=False):
                st.markdown(ml_report.read_text(encoding="utf-8"))
