"""Stage A Robometer worker loop.

Two-phase execution:
  Phase 1 — drain up to max_episodes from queue, fetch MP4 bytes, preprocess
             (decode → center-crop → 256×256 → 2 fps) in a thread pool.
  Phase 2 — score all preprocessed frames sequentially on the GPU.

Separating phases means the GPU is never idle waiting for decoding, and
decoding is never blocked waiting for the GPU.
"""

from __future__ import annotations

import os
import signal
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def prefetch_episodes(
    queue: WorkQueue,
    store: ObjectStore,
    catalog: Catalog,
    model_version: str,
    max_episodes: int | None = None,
    n_preprocess_workers: int = 8,
) -> tuple[list[tuple], list]:
    """Phase 1: drain queue, fetch MP4s, preprocess in parallel.

    Returns (preprocessed, skipped_receipts) where preprocessed is a list of
    (msg, manifest, frames, error) tuples ordered by queue position.
    Safe to call before the Robometer server is ready.
    """
    drained: list[tuple] = []
    skipped_receipts: list = []
    limit = max_episodes if max_episodes is not None else float("inf")

    logger.info("Phase 1: draining queue", limit=max_episodes)
    while len(drained) + len(skipped_receipts) < limit:
        msgs = queue.receive(max_messages=1, wait_seconds=1)
        if not msgs:
            break
        msg = msgs[0]
        try:
            manifest = EpisodeManifest.model_validate(msg.body)
        except Exception as exc:
            logger.error("Failed to parse EpisodeManifest", error=str(exc))
            queue.send_to_dlq(msg.body)
            queue.delete(msg.receipt)
            continue

        if already_scored_robometer(catalog, manifest.episode_id, model_version):
            logger.info("Skipping — already scored", episode_id=manifest.episode_id)
            skipped_receipts.append(msg.receipt)
            continue

        try:
            mp4_bytes = store.get_bytes(manifest.head_video_key)
        except Exception as exc:
            logger.error("Failed to fetch MP4", episode_id=manifest.episode_id, error=str(exc))
            queue.send_to_dlq(msg.body)
            queue.delete(msg.receipt)
            continue

        drained.append((msg, manifest, mp4_bytes))

    def _preprocess(item: tuple) -> tuple:
        msg, manifest, mp4_bytes = item
        try:
            return msg, manifest, preprocess_video(mp4_bytes), None
        except Exception as exc:
            return msg, manifest, None, str(exc)

    logger.info("Phase 1: preprocessing", n_episodes=len(drained), workers=n_preprocess_workers)
    t_pre = time.monotonic()
    preprocessed: list[tuple] = [None] * len(drained)
    with ThreadPoolExecutor(max_workers=n_preprocess_workers) as pool:
        futures = {pool.submit(_preprocess, item): i for i, item in enumerate(drained)}
        for future in as_completed(futures):
            preprocessed[futures[future]] = future.result()
    pre_ms = (time.monotonic() - t_pre) * 1000
    logger.info("Phase 1 complete", n_episodes=len(preprocessed), preprocess_ms=round(pre_ms, 1))

    return preprocessed, skipped_receipts


def run_worker(
    model: RobometerModel | None = None,
    queue: WorkQueue | None = None,
    store: ObjectStore | None = None,
    catalog: Catalog | None = None,
    robometer_threshold: float = 0.5,
    poll_wait: int = 5,
    max_episodes: int | None = None,
    n_preprocess_workers: int = 8,
    prefetched: tuple | None = None,
) -> int:
    """Two-phase worker. Returns number of episodes processed.

    Pass prefetched=(preprocessed, skipped_receipts) from prefetch_episodes()
    to skip Phase 1 (e.g. when decode ran concurrently with server startup).
    """
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

    model_version = getattr(model, "model_version", "unknown")
    logger.info("Stage A Robometer worker started", model_version=model_version)

    # ------------------------------------------------------------------ #
    # Phase 1: drain queue → fetch bytes → parallel preprocess            #
    # ------------------------------------------------------------------ #
    if prefetched is not None:
        preprocessed, skipped_receipts = prefetched
        logger.info("Phase 1: using pre-fetched frames", n_episodes=len(preprocessed))
    else:
        preprocessed, skipped_receipts = prefetch_episodes(
            queue=queue,
            store=store,
            catalog=catalog,
            model_version=model_version,
            max_episodes=max_episodes,
            n_preprocess_workers=n_preprocess_workers,
        )

    # Clean up already-scored queue entries
    for receipt in skipped_receipts:
        queue.delete(receipt)

    # ------------------------------------------------------------------ #
    # Phase 2: score sequentially (GPU)                                   #
    # ------------------------------------------------------------------ #
    logger.info("Phase 2: scoring", n_episodes=len(preprocessed))

    processed = len(skipped_receipts)

    for msg, manifest, frames, preprocess_error in preprocessed:
        clear_context()
        bind_episode(manifest.episode_id, task=manifest.task)

        if preprocess_error is not None:
            logger.error("Preprocessing failed — sending to DLQ", error=preprocess_error)
            queue.send_to_dlq(msg.body)
            queue.delete(msg.receipt)
            processed += 1
            continue

        try:
            t0 = time.monotonic()

            score: RobometerScore
            if hasattr(model, "score_episode_from_frames"):
                score = model.score_episode_from_frames(frames, manifest.instruction)
            elif hasattr(model, "score_episode_from_bytes"):
                mp4_bytes = store.get_bytes(manifest.head_video_key)
                score = model.score_episode_from_bytes(mp4_bytes, manifest.instruction)
            else:
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
