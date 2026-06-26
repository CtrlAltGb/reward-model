"""Structured JSON logging via structlog."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    return structlog.get_logger(name)


def bind_episode(episode_id: str, **kw) -> None:
    structlog.contextvars.bind_contextvars(episode_id=episode_id, **kw)


def bind_cohort(cohort_id: str, **kw) -> None:
    structlog.contextvars.bind_contextvars(cohort_id=cohort_id, **kw)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
