"""Agents + Crew — everything wired together."""

from crewai import Agent, Crew, Process
from crewai_tools import TavilySearchTool
from config.settings import llm
from tools.tasks import orchestrate_task, eda_task, ml_task
from prompts.orchestrator_system import SYSTEM_PROMPT as ORCHESTRATOR_PROMPT
from prompts.eda_system import SYSTEM_PROMPT as EDA_PROMPT
from prompts.ml_system import SYSTEM_PROMPT as ML_PROMPT

# ── Agents ──────────────────────────────────────────────

search_tool = TavilySearchTool()

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

crew = Crew(
    agents=[orchestrator, eda_agent, ml_agent],
    tasks=[orchestrate_task, eda_task, ml_task],
    process=Process.sequential,
    verbose=True,
)
