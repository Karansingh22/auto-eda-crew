"""Task definitions."""

from crewai import Task


route_task = Task(
    description=(
        "Decide how to handle this user input: '{topic}'. Use semantic understanding, "
        "not keyword matching. Return ANALYSIS only when the user clearly wants full "
        "EDA, supervised ML modeling, feature engineering, model evaluation, or a report "
        "based on the provided DATASET CONTEXT. Return CHAT for greetings, capability "
        "questions, normal conversation, missing-dataset guidance, or general questions. "
        "Return exactly one word: CHAT or ANALYSIS. Do not explain your choice."
    ),
    expected_output="Exactly one word: CHAT or ANALYSIS.",
)

orchestrate_task = Task(
    description=(
        "Handle the user input: '{topic}'. For greetings or capability questions, respond "
        "briefly and explain the dataset workflow. For general questions, answer directly "
        "and search only when current information is needed. For dataset analysis requests, "
        "summarize the dataset context and the user's goal for downstream agents. Keep the "
        "response under 180 words unless the user asks for detail."
    ),
    expected_output="A concise, helpful response under 180 words or a clear dataset analysis handoff.",
)

eda_task = Task(
    description=(
        "Perform exploratory data analysis only when '{topic}' includes DATASET CONTEXT. "
        "Use the provided profile to cover shape, dtypes, missing values, duplicates, "
        "numeric statistics, categorical distributions, target behavior, correlations or "
        "correlation limitations, data quality risks, and practical next steps. Keep it "
        "concise and prioritize the top issues."
    ),
    expected_output="A concise structured EDA report with the top actionable recommendations.",
    output_file="output/eda_report.md",
)

ml_task = Task(
    description=(
        "Provide ML analysis only when '{topic}' includes DATASET CONTEXT. Use the target "
        "variable, feature types, missingness, imbalance signals, and sample rows to infer "
        "the supervised task type and recommend problem framing, preprocessing, feature "
        "engineering, model choices, evaluation metrics, validation strategy, and deployment "
        "risks. Use Tavily research when current or domain-specific algorithm guidance would "
        "make the recommendation stronger. Keep it concise and prioritize practical baselines."
    ),
    expected_output="A concise structured ML report with practical model recommendations.",
    output_file="output/ml_report.md",
)

vis_task = Task(
    description=(
        "Generate a standalone, completely valid python script to create beautiful, interactive Plotly charts based on the dataset profile: '{topic}'. "
        "The script MUST define a function `generate_plots(df)` that takes a pandas DataFrame `df` and returns a list of dictionaries. "
        "Inside `generate_plots`, detect the column types and target variable, create 3-4 highly tailored interactive Plotly figures "
        "(e.g., target distribution, correlation matrix heatmap, scatter plots for numeric features against target, or box/bar plots for categorical features), "
        "convert each figure to a JSON-compatible dictionary using `fig.to_plotly_json()` (or `plotly.io.to_json(fig)` but to_plotly_json() is preferred), "
        "and save the list of these dictionaries to a JSON file located at `output/plotly_charts.json`. "
        "The generated script should be completely self-contained, import pandas, plotly.express, plotly.graph_objects, and json, "
        "handle missing values or large categories gracefully (e.g. limiting categorical plots to top 10 categories), "
        "and contain robust try/except blocks inside `generate_plots`. "
        "The generated Python script must be saved directly to `output/vis_code.py`. "
        "Make sure to output ONLY valid Python code in the file, with NO markdown formatting, NO backticks (```python), and NO extra explanations."
    ),
    expected_output="A valid, self-contained Python script saved to output/vis_code.py that generates Plotly charts JSON.",
    output_file="output/vis_code.py",
)

pipeline_task = Task(
    description=(
        "Based on the dataset profile and ML task analysis for '{topic}', generate a complete, production-ready AutoML training script in Python. "
        "The script must perform target-aware data cleaning, robust preprocessing (scaling, imputation, one-hot encoding for tabular columns), "
        "stratified K-fold cross-validation (or time-aware K-fold if a temporal pattern is found), model training (Logistic/Linear Regression, "
        "Random Forest, or XGBoost depending on class/regression type), evaluate performance metrics, plot or save evaluation metrics, "
        "and save the final trained pipeline using `joblib`. "
        "The generated Python script must be saved directly to `output/pipeline.py`. "
        "Output ONLY valid Python code in the file, with NO markdown formatting, NO backticks (```python), and NO extra explanations."
    ),
    expected_output="A complete, executable Python training script saved to output/pipeline.py.",
    output_file="output/pipeline.py",
)

