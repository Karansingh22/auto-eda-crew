"""Helpers for running CrewAI with practical retry behavior."""

from __future__ import annotations

import re
import time
from typing import Any


_TRY_AGAIN_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


def kickoff_with_retry(crew: Any, *, inputs: dict[str, Any], retries: int = 2) -> Any:
    """Run a crew and wait through short Groq rate-limit windows."""

    for attempt in range(retries + 1):
        try:
            return crew.kickoff(inputs=inputs)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt >= retries:
                raise

            time.sleep(_retry_delay_seconds(exc))

    raise RuntimeError("Crew kickoff failed after retry attempts.")


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "ratelimiterror" in text or "rate limit" in text or "rate_limit_exceeded" in text


def _retry_delay_seconds(exc: Exception) -> float:
    match = _TRY_AGAIN_RE.search(str(exc))
    if match:
        return float(match.group(1)) + 1.0

    return 15.0
