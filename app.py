"""Streamlit UI for the CrewAI dataset analysis assistant."""

from pathlib import Path
import sys
import importlib.util
import json

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

# Inject premium custom CSS styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Outfit', sans-serif !important;
}
.hero-container {
    background: linear-gradient(135deg, rgba(16, 20, 30, 0.9) 0%, rgba(24, 30, 45, 0.9) 100%);
    padding: 2.5rem;
    border-radius: 16px;
    border: 1px solid rgba(0, 198, 255, 0.25);
    box-shadow: 0 8px 32px 0 rgba(0, 198, 255, 0.08);
    margin-bottom: 2rem;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #a0aec0;
    font-weight: 300;
    line-height: 1.6;
}
.glow-badge {
    background: rgba(0, 198, 255, 0.15);
    color: #00C6FF;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    border: 1px solid rgba(0, 198, 255, 0.3);
    display: inline-block;
    margin-bottom: 1rem;
}
</style>
""", unsafe_html=True)

# Render premium header
st.markdown("""
<div class="hero-container">
    <span class="glow-badge">⚡ Powered by CrewAI & Groq</span>
    <h1 class="hero-title">OmniAnalyst</h1>
    <p class="hero-subtitle">
        An advanced multi-agent system that delivers deep Exploratory Data Analysis, automated machine learning pipeline strategies, interactive Plotly dashboards, and production-ready downloadable training scripts with a single prompt.
    </p>
</div>
""", unsafe_html=True)

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
    
    st.divider()
    st.subheader("💡 Intelligent Insights & Assets")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 EDA & Analysis",
        "🤖 ML Strategy & Modeling",
        "🎨 Interactive AI Dashboard",
        "🚀 Production Pipeline Script"
    ])
    
    with tab1:
        if eda_report.exists():
            st.markdown(eda_report.read_text(encoding="utf-8"))
        else:
            st.info("EDA Report is being compiled or was not generated.")
            
    with tab2:
        if ml_report.exists():
            st.markdown(ml_report.read_text(encoding="utf-8"))
        else:
            st.info("ML Report is being compiled or was not generated.")
            
    with tab3:
        vis_script = OUTPUT_DIR / "vis_code.py"
        if vis_script.exists() and uploaded_file is not None:
            try:
                # Dynamically load the generated visualization script
                spec = importlib.util.spec_from_file_location("vis_code", str(vis_script))
                vis_module = importlib.util.module_from_spec(spec)
                sys.modules["vis_code"] = vis_module
                spec.loader.exec_module(vis_module)
                
                if hasattr(vis_module, "generate_plots"):
                    with st.spinner("Generating custom interactive visualizations from dataset..."):
                        plotly_dicts = vis_module.generate_plots(df)
                        
                        import plotly.graph_objects as go
                        
                        if plotly_dicts:
                            st.markdown("### 🎨 AI-Generated Interactive Dashboard")
                            st.caption("These interactive Plotly charts were custom-built by the agent specifically for your dataset and target variable.")
                            for i, chart in enumerate(plotly_dicts):
                                try:
                                    if isinstance(chart, dict):
                                        fig = go.Figure(chart)
                                    else:
                                        fig = chart
                                    
                                    fig.update_layout(
                                        template="plotly_dark",
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        margin=dict(l=20, r=20, t=40, b=20),
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.divider()
                                except Exception as chart_err:
                                    st.error(f"Error rendering chart {i+1}: {chart_err}")
                        else:
                            st.info("No interactive charts were generated by the visualization specialist.")
                else:
                    st.error("The generated visualization script does not define the expected `generate_plots(df)` function.")
            except Exception as vis_err:
                st.error(f"Failed to execute generated visualization code: {vis_err}")
        else:
            st.info("Upload a dataset and run analysis to generate the interactive AI dashboard.")
            
    with tab4:
        pipeline_file = OUTPUT_DIR / "pipeline.py"
        if pipeline_file.exists():
            pipeline_code = pipeline_file.read_text(encoding="utf-8")
            st.markdown("### 🚀 Download Ready-to-Run AutoML Pipeline")
            st.info("This production-ready Python script was custom-generated by the ML Specialist to preprocess, train, cross-validate, and export a calibrated model for your target variable.")
            st.code(pipeline_code, language="python")
            st.download_button(
                label="📥 Download pipeline.py",
                data=pipeline_code,
                file_name="pipeline.py",
                mime="text/x-python"
            )
        else:
            st.info("Upload a dataset and run analysis to generate the production pipeline script.")


