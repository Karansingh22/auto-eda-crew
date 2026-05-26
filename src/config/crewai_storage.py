"""CrewAI storage configuration."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def crewai_storage_path() -> Path:
    return Path(__file__).resolve().parents[2] / "output" / ".crewai_storage"


def configure_crewai_storage() -> None:
    """Keep CrewAI's SQLite storage inside this project workspace."""

    storage_root = crewai_storage_path()
    storage_root.mkdir(parents=True, exist_ok=True)

    # CrewAI uses appdirs on Windows, which writes under LOCALAPPDATA.
    # Point it to a project-local writable folder before importing crewai.
    os.environ["LOCALAPPDATA"] = str(storage_root)
    os.environ.setdefault("CREWAI_STORAGE_DIR", "CrewAI")


def patch_crewai_storage_path() -> None:
    """Patch CrewAI's task-output SQLite storage helper after import."""

    storage_root = crewai_storage_path()
    storage_root.mkdir(parents=True, exist_ok=True)

    def local_storage_path() -> str:
        return str(storage_root)

    module_names = (
        "crewai.utilities.paths",
        "crewai.memory.storage.kickoff_task_outputs_storage",
    )

    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "db_storage_path"):
            setattr(module, "db_storage_path", local_storage_path)
