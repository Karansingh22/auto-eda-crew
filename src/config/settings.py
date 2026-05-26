"""Centralised settings — loads .env and exposes LLM + app config."""

import os
from pathlib import Path
from dotenv import load_dotenv

from config.crewai_storage import configure_crewai_storage, patch_crewai_storage_path

configure_crewai_storage()

from crewai import LLM

patch_crewai_storage_path()

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# --- App settings ---
TOPIC: str = os.getenv("TOPIC", "AI Agents")
OUTPUT_DIR: Path = Path(__file__).resolve().parents[2] / "output"
LLM_MAX_TOKENS: int = min(
    int(os.getenv("LLM_MAX_TOKENS", "1536")),
    int(os.getenv("LLM_TOKEN_CAP", "1536")),
)

# --- Groq LLM ---
llm = LLM(
    model=os.getenv("MODEL", "groq/llama-3.3-70b-versatile"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.4")),
    max_tokens=LLM_MAX_TOKENS,
)
