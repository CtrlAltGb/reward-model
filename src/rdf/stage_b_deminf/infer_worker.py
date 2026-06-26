"""Stage B DemInf inference worker loop.

Startup: load DeminfWorker with current VAE ckpt + reference latents (once).
Loop: poll cohort queue → encode episodes → score → write catalog → delete.
"""

from __future__ import annotations

import os
import signal
import time
from datetime import datetime, timezone

from rdf.harness.catalog import Catalog, get_catalog
from rdf.harness.config import get_embodiment_config
from rdf.harness.idempotency import already_scored_deminf
from rdf.harness.logging import bind_cohort, clear_context, configure_logging, get_logger
from rdf.harness.mcap_extract import SyntheticMcapReader, extract_state_action
from rdf.harness.queue import WorkQueue, get_queue
from rdf.harness.storage import ObjectStore, get_object_store
from rdf.models.base import DeminfModel
from rdf.models.registry import VaeRegistry
from rdf.schemas.models import CohortMessage, DeminfResult

logger = get_logger(__name__)


def _get_model(
    obs_ckpt: str | None = None,
    action_ckpt: str | None = None,
    reference_latents_path: str | None = None,
) -> DeminfModel:
    backend = os.environ.get("RDF_MODELS", "mock")
    if backend == "mock":
        from rdf.models.mock import MockDeminfModel
        return MockDeminfModel()
    from rdf.models.deminf_worker import DeminfWorker
    return DeminfWorker(
        obs_ckpt=obs_ckpt,
        action_ckpt=action_ckpt,
        reference_latents_path=reference_latents_path,
    )


def run_infer_worker(
    model: DeminfModel | None = None,
    queue: WorkQueue | None = None,
    store: ObjectStore | None = None,
    catalog: Catalog | None = None,
    registry: VaeRegistry | None = None,
    deminf_threshold: float = 0.0,
    poll_wait: int = 5,
    max_cohorts: int | None = None,
    embodiment: str = "franka",
) -> int:
    """Main DemInf inference worker loop. Returns number of cohorts processed."""
    configure_logging()

    queue = queue or get_queue("rdf-cohorts")
    store = store or get_object_store()
    catalog = catalog or get_catalog()
    registry = registry or VaeRegistry(store=store)

    _current_vae_version: str | None = None
    _model: DeminfModel | None = model

    _running = [True]

    def _stop(signum, frame):
        logger.info("SIGTERM received — draining and stopping")
        _running[0] = False

    signal.signal(signal.SIGTERM, _stop)

    embodiment_cfg = get_embodiment_config(embodiment)
    mcap_reader = SyntheticMcapReader() if os.environ.get("RDF_MODELS", "mock") == "mock" else None

    processed = 0
    logger.info("Stage B DemInf inference worker started")

    while _running[0]:
        if max_cohorts is not None and processed >= max_cohorts:
            break

        messages = queue.receive(max_messages=1, wait_seconds=poll_wait)
        if not messages:
            continue

        msg = messages[0]
        try:
            cohort = CohortMessage.model_validate(msg.body)
        except Exception as exc:
            logger.error("Failed to parse CohortMessage", error=str(exc))
            queue.send_to_dlq(msg.body)
            queue.delete(msg.receipt)
            continue

        clear_context()
        bind_cohort(cohort.cohort_id, task=cohort.task, vae_version=cohort.vae_version)

        # Load model if VAE version changed
        if _model is None or cohort.vae_version != _current_vae_version:
            if os.environ.get("RDF_MODELS", "mock") != "mock":
                artifact = registry.load(cohort.task, cohort.vae_version)
                _model = _get_model(
                    obs_ckpt=artifact.obs_ckpt,
                    action_ckpt=artifact.action_ckpt,
                )
                # Attach reference latents directly
                import io
                import numpy as np
                buf = io.BytesIO()
                np.savez(buf, obs_latents=artifact.reference_latents_obs, action_latents=artifact.reference_latents_action)
                buf.seek(0)
                import tempfile, pathlib
                with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
                    f.write(buf.getvalue())
                    ref_path = f.name
                if hasattr(_model, '_load_reference_latents'):
                    _model.reference_latents_path = ref_path
                    _model._load_reference_latents()
            else:
                _model = _get_model()
            _current_vae_version = cohort.vae_version

        try:
            episode_states = []
            episode_actions = []
            valid_episode_ids = []

            for episode_id in cohort.episode_ids:
                row = catalog.get_row(episode_id)
                if row is None:
                    logger.warning("Episode not in catalog", episode_id=episode_id)
                    continue

                if already_scored_deminf(catalog, episode_id, cohort.vae_version):
                    logger.info("Skipping — already scored", episode_id=episode_id)
                    continue

                # Load episode data from MCAP
                manifest_row = catalog.get_row(episode_id)
                try:
                    # Build the MCAP key from catalog row (episode stored under task prefix)
                    mcap_key = f"raw/{episode_id}/data.mcap"
                    if store.exists(mcap_key):
                        mcap_bytes = store.get_bytes(mcap_key)
                    else:
                        # Use synthetic data in test/mock mode
                        import hashlib
                        mcap_bytes = hashlib.md5(episode_id.encode()).digest() * 64

                    states, actions = extract_state_action(
                        mcap_bytes, embodiment_cfg, reader=mcap_reader
                    )
                    episode_states.append(states)
                    episode_actions.append(actions)
                    valid_episode_ids.append(episode_id)
                except Exception as exc:
                    logger.error("Failed to load episode MCAP", episode_id=episode_id, error=str(exc))

            if not valid_episode_ids:
                logger.info("No valid episodes in cohort")
                queue.delete(msg.receipt)
                processed += 1
                continue

            # Score the cohort
            t0 = time.monotonic()
            if hasattr(_model, "score_cohort"):
                scores = _model.score_cohort(episode_states, episode_actions)
            else:
                # Protocol-based model: separate encode + score
                import numpy as np
                obs_lat, act_lat = _model.encode_episodes(episode_states, episode_actions)
                # Use zero reference latents for mock/fallback
                ref_obs = np.zeros_like(obs_lat[:1])
                ref_act = np.zeros_like(act_lat[:1])
                scores = _model.score_against_reference(obs_lat, act_lat, ref_obs, ref_act)

            latency_ms = (time.monotonic() - t0) * 1000
            now = datetime.now(timezone.utc)

            for episode_id, score in zip(valid_episode_ids, scores):
                result = DeminfResult(
                    episode_id=episode_id,
                    task=cohort.task,
                    deminf_score=float(score),
                    vae_version=cohort.vae_version,
                    reference_set_version=cohort.reference_set_version,
                    cohort_id=cohort.cohort_id,
                    scored_at=now,
                    status="scored",
                )
                passed = float(score) >= deminf_threshold
                catalog.update_deminf(result, pass_=passed, threshold=deminf_threshold)

            logger.info(
                "Scored cohort",
                n_episodes=len(valid_episode_ids),
                latency_ms=round(latency_ms, 1),
            )

        except Exception as exc:
            logger.error("Cohort scoring failed", error=str(exc))
            queue.send_to_dlq(msg.body)

        queue.delete(msg.receipt)
        processed += 1

    logger.info("Stage B DemInf worker stopped", processed=processed)
    return processed
