"""Materializer — copy kept episodes from raw bucket to clean bucket.

Idempotent: re-running produces the same result.
Output layout: clean/task=<task>/episode=<episode_id>/{head.mp4, data.mcap, metadata.yaml}
"""

from __future__ import annotations

from rdf.harness.catalog import Catalog
from rdf.harness.logging import get_logger
from rdf.harness.storage import ObjectStore

logger = get_logger(__name__)

_EPISODE_FILES = ["head.mp4", "data.mcap", "metadata.yaml"]


def materialize_task(
    task: str,
    catalog: Catalog,
    raw_store: ObjectStore,
    clean_store: ObjectStore,
) -> dict[str, int]:
    """Copy all kept episodes for a task to the clean bucket.

    Returns counts: {copied, skipped, failed}
    """
    rows = catalog.rows_for_task(task)
    kept = [r for r in rows if r.final_decision == "keep"]

    counts = {"copied": 0, "skipped": 0, "failed": 0}

    for row in kept:
        ep_id = row.episode_id
        for filename in _EPISODE_FILES:
            raw_key = f"raw/{ep_id}/{filename}"
            clean_key = f"clean/task={task}/episode={ep_id}/{filename}"

            if clean_store.exists(clean_key):
                counts["skipped"] += 1
                continue

            if not raw_store.exists(raw_key):
                logger.warning("Raw file missing", episode_id=ep_id, key=raw_key)
                counts["failed"] += 1
                continue

            try:
                data = raw_store.get_bytes(raw_key)
                clean_store.put_bytes(clean_key, data)
                counts["copied"] += 1
            except Exception as exc:
                logger.error("Failed to copy", episode_id=ep_id, key=raw_key, error=str(exc))
                counts["failed"] += 1

    logger.info("Materialization complete", task=task, **counts)
    return counts
