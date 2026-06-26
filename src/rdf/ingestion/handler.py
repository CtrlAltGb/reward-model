"""Ingestion handler — triggered by S3 PUT on *.mp4.

Checks all three files exist → parses metadata.yaml → builds EpisodeManifest → enqueues.
Idempotent on recording_id.
"""

from __future__ import annotations

from datetime import datetime, timezone

import yaml

from rdf.harness.logging import get_logger
from rdf.harness.queue import WorkQueue
from rdf.harness.storage import ObjectStore
from rdf.schemas.models import EpisodeManifest

logger = get_logger(__name__)

_REQUIRED_SUFFIXES = ["head.mp4", "data.mcap", "metadata.yaml"]


def handle_s3_put(
    object_key: str,
    store: ObjectStore,
    episode_queue: WorkQueue,
) -> EpisodeManifest | None:
    """Handle an S3 PUT event for a new episode file.

    Expects objects under: raw/<episode_id>/{head.mp4, data.mcap, metadata.yaml}
    """
    # Extract episode_id from key
    parts = object_key.split("/")
    if len(parts) < 3 or parts[0] != "raw":
        logger.warning("Unexpected S3 key format", key=object_key)
        return None

    episode_id = parts[1]
    prefix = f"raw/{episode_id}"

    # Check all required files exist
    for suffix in _REQUIRED_SUFFIXES:
        key = f"{prefix}/{suffix}"
        if not store.exists(key):
            logger.info("Waiting for sibling files", episode_id=episode_id, missing=key)
            return None

    # Parse metadata
    try:
        meta_bytes = store.get_bytes(f"{prefix}/metadata.yaml")
        meta = yaml.safe_load(meta_bytes)
    except Exception as exc:
        logger.error("Failed to parse metadata.yaml", episode_id=episode_id, error=str(exc))
        return None

    manifest = EpisodeManifest(
        episode_id=episode_id,
        s3_prefix=prefix + "/",
        robot_id=meta.get("robot_id", "unknown"),
        embodiment=meta.get("embodiment", "franka"),
        task=meta.get("task", "unknown"),
        instruction=meta.get("instruction", ""),
        head_video_key=f"{prefix}/head.mp4",
        mcap_key=f"{prefix}/data.mcap",
        metadata_key=f"{prefix}/metadata.yaml",
        created_at=datetime.fromisoformat(meta["created_at"]) if "created_at" in meta
        else datetime.now(timezone.utc),
    )

    episode_queue.enqueue(
        manifest.model_dump(mode="json"),
        dedup_id=episode_id,
    )
    logger.info("Enqueued episode", episode_id=episode_id, task=manifest.task)
    return manifest
