"""Agents + Crew — everything wired together."""

from crewai import Agent, Crew, Process
from crewai_tools import TavilySearchTool
from config.settings import llm
from tools.tasks import (
    route_task,
    orchestrate_task,
    eda_task,
    ml_task,
    vis_task,
    pipeline_task,
)
from prompts.orchestrator_system import SYSTEM_PROMPT as ORCHESTRATOR_PROMPT
from prompts.eda_system import SYSTEM_PROMPT as EDA_PROMPT
from prompts.ml_system import SYSTEM_PROMPT as ML_PROMPT

# ── Tools ───────────────────────────────────────────────

search_tool = TavilySearchTool()

# ── Agents ──────────────────────────────────────────────

router = Agent(
    role="Router",
    goal="Determine the user's intent with extreme accuracy. Decide if they want a general greeting/chat or a detailed dataset analysis.",
    backstory="Highly analytical query classifier. Decides instantly and correctly.",
    llm=llm,
    allow_delegation=False,
)

orchestrator = Agent(
    role="Orchestrator",
    goal="Handle greetings, answer general questions, perform web searches",
    backstory="Friendly assistant who greets users, answers queries, "
              "and searches the web when needed.",
    llm=llm,
    tools=[search_tool],
    allow_delegation=True,
    system_template=ORCHESTRATOR_PROMPT + "\n\n{{ .System }}",
)

eda_agent = Agent(
    role="EDA Specialist",
    goal="Perform thorough exploratory data analysis on datasets",
    backstory="Senior data analyst with deep expertise in EDA, "
              "statistics, and data quality assessment.",
    llm=llm,
    system_template=EDA_PROMPT + "\n\n{{ .System }}",
)

ml_agent = Agent(
    role="ML Analyst",
    goal="Design and recommend machine learning solutions for data problems",
    backstory="ML engineer experienced in model selection, "
              "feature engineering, and production pipelines.",
    llm=llm,
    system_template=ML_PROMPT + "\n\n{{ .System }}",
)

vis_agent = Agent(
    role="Visualization Specialist",
    goal="Create beautiful, modern, and highly insightful interactive Plotly charts in Python.",
    backstory="Expert data visualizer who understands which charts best reveal patterns in numerical, categorical, and target variables.",
    llm=llm,
)

# Assign agents to tasks to satisfy CrewAI's validation requirements
route_task.agent = router
orchestrate_task.agent = orchestrator
eda_task.agent = eda_agent
ml_task.agent = ml_agent
vis_task.agent = vis_agent
pipeline_task.agent = ml_agent

# ── Crews ───────────────────────────────────────────────

route_crew = Crew(
    agents=[router],
    tasks=[route_task],
    process=Process.sequential,
    verbose=True,
)

chat_crew = Crew(
    agents=[orchestrator],
    tasks=[orchestrate_task],
    process=Process.sequential,
    verbose=True,
)

analysis_crew = Crew(
    agents=[eda_agent, ml_agent, vis_agent],
    tasks=[eda_task, ml_task, vis_task, pipeline_task],
    process=Process.sequential,
    verbose=True,
)
