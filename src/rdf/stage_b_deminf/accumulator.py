"""Cohort accumulator — groups passed episodes into cohorts for Stage B.

Sequential mode: waits for all Stage A to finish, emits one cohort per task.
Parallel mode: fires when cohort_min episodes accumulate or timeout elapses.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from rdf.harness.catalog import Catalog
from rdf.harness.logging import get_logger
from rdf.harness.queue import WorkQueue
from rdf.schemas.models import CohortMessage, PipelineConfig

logger = get_logger(__name__)


def _passed_episode_ids(catalog: Catalog, task: str) -> list[str]:
    rows = catalog.rows_for_task(task)
    return [r.episode_id for r in rows if r.robometer_pass is True]


def accumulate_sequential(
    tasks: list[str],
    catalog: Catalog,
    cohort_queue: WorkQueue,
    pipeline_cfg: PipelineConfig,
    vae_version: str,
    reference_set_version: str,
) -> list[str]:
    """Wait for all episodes to be scored, then emit one cohort per task."""
    cohort_ids = []
    for task in tasks:
        episode_ids = _passed_episode_ids(catalog, task)
        if not episode_ids:
            logger.info("No passed episodes for task", task=task)
            continue

        cohort_id = str(uuid.uuid4())
        msg = CohortMessage(
            cohort_id=cohort_id,
            task=task,
            episode_ids=episode_ids,
            vae_version=vae_version,
            reference_set_version=reference_set_version,
            created_at=datetime.now(timezone.utc),
        )
        cohort_queue.enqueue(msg.model_dump(mode="json"), dedup_id=f"{task}-sequential")
        cohort_ids.append(cohort_id)
        logger.info("Emitted cohort", task=task, n_episodes=len(episode_ids), cohort_id=cohort_id)

    return cohort_ids


def accumulate_parallel(
    task: str,
    catalog: Catalog,
    cohort_queue: WorkQueue,
    pipeline_cfg: PipelineConfig,
    vae_version: str,
    reference_set_version: str,
    deadline: float | None = None,
) -> str | None:
    """Poll until cohort_min passed episodes accumulate or timeout elapses."""
    deadline = deadline or (time.time() + pipeline_cfg.deminf_cohort_timeout_hours * 3600)

    while time.time() < deadline:
        episode_ids = _passed_episode_ids(catalog, task)
        n = len(episode_ids)

        if n >= pipeline_cfg.deminf_cohort_min:
            batch = episode_ids[: pipeline_cfg.deminf_cohort_max]
            cohort_id = str(uuid.uuid4())
            msg = CohortMessage(
                cohort_id=cohort_id,
                task=task,
                episode_ids=batch,
                vae_version=vae_version,
                reference_set_version=reference_set_version,
                created_at=datetime.now(timezone.utc),
            )
            cohort_queue.enqueue(msg.model_dump(mode="json"), dedup_id=f"{task}-{cohort_id}")
            logger.info("Emitted parallel cohort", task=task, n_episodes=len(batch))
            return cohort_id

        logger.info("Waiting for cohort_min", task=task, have=n, need=pipeline_cfg.deminf_cohort_min)
        time.sleep(30)

    logger.warning("Cohort timeout — emitting partial cohort", task=task)
    episode_ids = _passed_episode_ids(catalog, task)
    if not episode_ids:
        return None
    cohort_id = str(uuid.uuid4())
    msg = CohortMessage(
        cohort_id=cohort_id,
        task=task,
        episode_ids=episode_ids[: pipeline_cfg.deminf_cohort_max],
        vae_version=vae_version,
        reference_set_version=reference_set_version,
        created_at=datetime.now(timezone.utc),
    )
    cohort_queue.enqueue(msg.model_dump(mode="json"))
    return cohort_id
