# CrewAI + Groq Research Crew

This is a small CrewAI project wired for Groq. It follows the standard CrewAI crew layout:

```text
src/crewai_groq_research/
  main.py
  crew.py
  config/
    agents.yaml
    tasks.yaml
```

## Setup

1. Install Python `>=3.10,<3.14`.
2. Install or update the CrewAI CLI if needed:

```powershell
uv tool install crewai --upgrade
```

3. Create your local `.env` from `.env.example` and set your real Groq key:

```powershell
Copy-Item .env.example .env
```

4. Install project dependencies:

```powershell
crewai install
```

5. Run the crew:

```powershell
crewai run
```

The final report is written to `output/report.md`.

## Groq Configuration

The default model is set in `.env.example`:

```dotenv
MODEL=groq/llama-3.3-70b-versatile
```

CrewAI routes Groq through LiteLLM, so this project depends on `crewai[litellm]`. You can switch models by changing `MODEL`, as long as the value follows CrewAI/LiteLLM provider format such as `groq/<model-id>`.

## Customizing The Topic

Change the `TOPIC` value in `.env`, or edit `src/crewai_groq_research/main.py` if you want to pass inputs from another app.
