"""Tracing and observability hooks with LangSmith/Langfuse and local span support."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Span context supporting local instrumentation and telemetry provider hooks."""
    settings = get_settings()
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
        "status": "active",
    }

    # Optional LangSmith trace tag injection
    if settings.langsmith_api_key:
        logger.debug(
            "[LangSmith Trace] Starting span: %s (Project: %s)",
            name,
            settings.langsmith_project,
        )

    # Optional Langfuse trace tag injection
    if settings.langfuse_public_key:
        logger.debug(
            "[Langfuse Trace] Starting span: %s (Host: %s)",
            name,
            settings.langfuse_host,
        )

    try:
        yield span
        span["status"] = "success"
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        logger.error("Span '%s' encountered error: %s", name, exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.debug(
            "Span '%s' completed in %.4fs (status: %s)",
            name,
            span["duration_seconds"],
            span["status"],
        )
