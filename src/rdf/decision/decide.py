"""Pure decision function — no model calls.

Applies robometer gate then deminf trim; writes final_decision to catalog.
"""

from __future__ import annotations

from rdf.harness.catalog import Catalog
from rdf.harness.logging import get_logger

logger = get_logger(__name__)


def decide_episode(
    episode_id: str,
    catalog: Catalog,
    robometer_threshold: float,
    deminf_threshold: float,
) -> str:
    """Compute and persist final decision for one episode. Returns decision string."""
    row = catalog.get_row(episode_id)
    if row is None:
        raise KeyError(f"Episode {episode_id!r} not found in catalog")

    if row.final_decision != "pending":
        return row.final_decision

    reasons: list[str] = []

    # Stage A gate
    if row.robometer_success_pred is None:
        reasons.append("robometer_not_scored")
        catalog.finalize(episode_id, "pending", reasons)
        return "pending"

    if row.robometer_success_pred < robometer_threshold:
        reasons.append("task_incomplete")
        catalog.finalize(episode_id, "drop", reasons)
        logger.info("Decision: drop (task_incomplete)", episode_id=episode_id)
        return "drop"

    # Stage B trim
    if row.deminf_score is None:
        reasons.append("deminf_not_scored")
        catalog.finalize(episode_id, "pending", reasons)
        return "pending"

    if row.deminf_score < deminf_threshold:
        reasons.append("low_quality_jitter")
        catalog.finalize(episode_id, "drop", reasons)
        logger.info("Decision: drop (low_quality_jitter)", episode_id=episode_id)
        return "drop"

    catalog.finalize(episode_id, "keep", reasons)
    logger.info("Decision: keep", episode_id=episode_id)
    return "keep"


def decide_all(
    task: str,
    catalog: Catalog,
    robometer_threshold: float,
    deminf_threshold: float,
) -> dict[str, int]:
    """Decide all episodes for a task. Returns counts per decision."""
    rows = catalog.rows_for_task(task)
    counts: dict[str, int] = {"keep": 0, "drop": 0, "pending": 0}
    for row in rows:
        decision = decide_episode(
            row.episode_id, catalog, robometer_threshold, deminf_threshold
        )
        counts[decision] = counts.get(decision, 0) + 1
    return counts
