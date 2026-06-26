"""Stage A Robometer worker loop.

Startup: instantiate model (loads once).
Loop: poll episode queue → idempotency check → score → write catalog → delete.
"""

from __future__ import annotations

import os
import signal
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from rdf.harness.catalog import Catalog, get_catalog
from rdf.harness.idempotency import already_scored_robometer
from rdf.harness.logging import bind_episode, clear_context, configure_logging, get_logger
from rdf.harness.queue import WorkQueue, get_queue
from rdf.harness.storage import ObjectStore, get_object_store
from rdf.harness.video import preprocess_video
from rdf.models.base import RobometerModel, RobometerScore
from rdf.schemas.models import EpisodeManifest, RobometerResult

logger = get_logger(__name__)


def _get_model() -> RobometerModel:
    backend = os.environ.get("RDF_MODELS", "mock")
    if backend == "mock":
        from rdf.models.mock import MockRobometerModel
        return MockRobometerModel()
    from rdf.models.robometer_worker import RobometerWorker
    return RobometerWorker()


def run_worker(
    model: RobometerModel | None = None,
    queue: WorkQueue | None = None,
    store: ObjectStore | None = None,
    catalog: Catalog | None = None,
    robometer_threshold: float = 0.5,
    poll_wait: int = 5,
    max_episodes: int | None = None,
) -> int:
    """Main worker loop. Returns number of episodes processed."""
    configure_logging()

    model = model or _get_model()
    queue = queue or get_queue("rdf-episodes")
    store = store or get_object_store()
    catalog = catalog or get_catalog()

    _running = [True]

    def _stop(signum, frame):
        logger.info("SIGTERM received — draining and stopping")
        _running[0] = False

    signal.signal(signal.SIGTERM, _stop)

    processed = 0
    logger.info("Stage A Robometer worker started", model_version=getattr(model, "model_version", "?"))

    while _running[0]:
        if max_episodes is not None and processed >= max_episodes:
            break

        messages = queue.receive(max_messages=1, wait_seconds=poll_wait)
        if not messages:
            continue

        msg = messages[0]
        try:
            manifest = EpisodeManifest.model_validate(msg.body)
        except Exception as exc:
            logger.error("Failed to parse EpisodeManifest", error=str(exc))
            queue.send_to_dlq(msg.body)
            queue.delete(msg.receipt)
            processed += 1
            continue

        clear_context()
        bind_episode(manifest.episode_id, task=manifest.task)

        model_version = getattr(model, "model_version", "unknown")

        if already_scored_robometer(catalog, manifest.episode_id, model_version):
            logger.info("Skipping — already scored", idempotency=True)
            queue.delete(msg.receipt)
            processed += 1
            continue

        try:
            t0 = time.monotonic()
            mp4_bytes = store.get_bytes(manifest.head_video_key)

            # Preprocess: center-crop → 256×256 → 2 fps
            frames = preprocess_video(mp4_bytes)

            score: RobometerScore
            if hasattr(model, "score_episode_from_frames"):
                score = model.score_episode_from_frames(frames, manifest.instruction)
            elif hasattr(model, "score_episode_from_bytes"):
                score = model.score_episode_from_bytes(mp4_bytes, manifest.instruction)
            else:
                # Write preprocessed frames as a temp .npy for models that need a path
                with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
                    import numpy as np
                    np.save(f, frames)
                    tmp = f.name
                try:
                    score = model.score_episode(tmp, manifest.instruction)
                finally:
                    Path(tmp).unlink(missing_ok=True)

            latency_ms = (time.monotonic() - t0) * 1000
            now = datetime.now(timezone.utc)

            result = RobometerResult(
                episode_id=manifest.episode_id,
                task=manifest.task,
                embodiment=manifest.embodiment,
                robometer_reward=score.reward,
                robometer_success_pred=score.success_pred,
                frames_used=score.frames_used,
                model_version=score.model_version,
                latency_ms=latency_ms,
                scored_at=now,
                status="scored",
            )

            passed = score.success_pred >= robometer_threshold
            catalog.update_robometer(result, pass_=passed, threshold=robometer_threshold)

            logger.info(
                "Scored episode",
                reward=score.reward,
                success_pred=score.success_pred,
                passed=passed,
                latency_ms=round(latency_ms, 1),
            )

        except Exception as exc:
            logger.error("Scoring failed — sending to DLQ", error=str(exc))
            now = datetime.now(timezone.utc)
            result = RobometerResult(
                episode_id=manifest.episode_id,
                task=manifest.task,
                embodiment=manifest.embodiment,
                robometer_reward=0.0,
                robometer_success_pred=0.0,
                frames_used=0,
                model_version=getattr(model, "model_version", "unknown"),
                latency_ms=0.0,
                scored_at=now,
                status="failed",
                error=str(exc),
            )
            queue.send_to_dlq(msg.body)

        queue.delete(msg.receipt)
        processed += 1

    logger.info("Stage A worker stopped", processed=processed)
    return processed
