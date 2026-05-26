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
