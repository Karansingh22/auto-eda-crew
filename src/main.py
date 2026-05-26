"""CLI runner."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app_logic import clear_generated_reports
from crew_runner import kickoff_with_retry


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


def run():
    OUTPUT_DIR.mkdir(exist_ok=True)
    topic = input("Enter your query: ").strip() or "Hello!"

    clear_generated_reports(OUTPUT_DIR)
    from agent.agents import chat_crew

    result = kickoff_with_retry(chat_crew, inputs={"topic": topic})
    print(result)


if __name__ == "__main__":
    run()
