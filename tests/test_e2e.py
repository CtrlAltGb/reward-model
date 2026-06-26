"""End-to-end tests — full pipeline in mock mode, both sequential and parallel.

make e2e → runs these tests.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from rdf.decision.decide import decide_all
from rdf.decision.materialize import materialize_task
from rdf.harness.catalog import LocalCatalog
from rdf.harness.queue import LocalQueue
from rdf.harness.storage import LocalObjectStore
from rdf.models.mock import MockDeminfModel, MockRobometerModel, reset_init_call_count
from rdf.models.registry import VaeArtifact, VaeRegistry
from rdf.schemas.models import CatalogRow, CohortMessage, PipelineConfig
from rdf.stage_a_robometer.worker import run_worker as run_stage_a
from rdf.stage_b_deminf.infer_worker import run_infer_worker as run_stage_b
from tests.fixtures.generate_episode import make_episode_manifest, synthetic_mp4_bytes

NOW = datetime.now(timezone.utc)
TASK = "pick_cup"
N_EPISODES = 10


def _setup(tmp_path):
    store = LocalObjectStore(root=str(tmp_path / "storage"))
    ep_queue = LocalQueue("episodes", root=str(tmp_path / "queues"))
    cohort_queue = LocalQueue("cohorts", root=str(tmp_path / "queues"))
    catalog = LocalCatalog(root=str(tmp_path / "catalog"))
    registry = VaeRegistry(store=store)
    return store, ep_queue, cohort_queue, catalog, registry


def _seed_episodes(store, ep_queue, catalog, n=N_EPISODES):
    episode_ids = []
    for i in range(n):
        ep_id = f"ep-{i:03d}"
        manifest = make_episode_manifest(ep_id, task=TASK)
        store.put_bytes(manifest.head_video_key, synthetic_mp4_bytes(ep_id))
        store.put_bytes(manifest.mcap_key, b"\x00" * 640)
        ep_queue.enqueue(manifest.model_dump(mode="json"), dedup_id=ep_id)
        catalog.upsert_row(CatalogRow(
            episode_id=ep_id, task=TASK, embodiment="franka", robot_id="r1",
            pipeline_mode="sequential", created_at=NOW, updated_at=NOW,
        ))
        episode_ids.append(ep_id)
    return episode_ids


def _seed_registry(registry, store):
    obs_lat = np.zeros((5, 16), dtype=np.float32)
    artifact = VaeArtifact(
        task=TASK, vae_version="v1",
        obs_ckpt="/tmp/obs", action_ckpt="/tmp/act",
        reference_latents_obs=obs_lat,
        reference_latents_action=obs_lat,
    )
    registry.publish(artifact)


def _emit_cohort(catalog, cohort_queue, episode_ids):
    passed = [
        ep_id for ep_id in episode_ids
        if (row := catalog.get_row(ep_id)) and row.robometer_pass is True
    ]
    if not passed:
        return None
    cohort_id = str(uuid.uuid4())
    msg = CohortMessage(
        cohort_id=cohort_id,
        task=TASK,
        episode_ids=passed,
        vae_version="v1",
        reference_set_version="v1",
        created_at=NOW,
    )
    cohort_queue.enqueue(msg.model_dump(mode="json"), dedup_id=cohort_id)
    return cohort_id


class TestE2ESequential:
    """Full sequential pipeline: Stage A → cohort → Stage B → decide → materialize."""

    def test_pipeline_sequential(self, tmp_path):
        store, ep_queue, cohort_queue, catalog, registry = _setup(tmp_path)
        reset_init_call_count()

        # Seed
        episode_ids = _seed_episodes(store, ep_queue, catalog)
        _seed_registry(registry, store)

        # Stage A
        rob_model = MockRobometerModel(seed=42)
        processed_a = run_stage_a(
            model=rob_model, queue=ep_queue, store=store, catalog=catalog,
            robometer_threshold=0.0,  # pass everything
            poll_wait=0, max_episodes=N_EPISODES,
        )
        assert processed_a == N_EPISODES
        assert ep_queue.depth() == 0

        # All episodes scored
        for ep_id in episode_ids:
            row = catalog.get_row(ep_id)
            assert row.robometer_reward is not None

        # Emit cohort
        cohort_id = _emit_cohort(catalog, cohort_queue, episode_ids)
        assert cohort_id is not None

        # Stage B
        dem_model = MockDeminfModel(seed=42)
        processed_b = run_stage_b(
            model=dem_model, queue=cohort_queue, store=store, catalog=catalog,
            registry=registry, deminf_threshold=-999.0,  # pass everything
            poll_wait=0, max_cohorts=1,
        )
        assert processed_b == 1

        # Decide
        counts = decide_all(TASK, catalog, 0.0, -999.0)
        assert counts["keep"] > 0
        assert counts["pending"] == 0

        # Materialize — seed raw files for kept episodes
        for ep_id in episode_ids:
            for fname in ["head.mp4", "data.mcap", "metadata.yaml"]:
                if not store.exists(f"raw/{ep_id}/{fname}"):
                    store.put_bytes(f"raw/{ep_id}/{fname}", b"data")

        clean_store = LocalObjectStore(root=str(tmp_path / "clean"))
        mat = materialize_task(TASK, catalog, store, clean_store)
        assert mat["copied"] > 0

    def test_robometer_gate_enforced(self, tmp_path):
        """Robometer gate must block episodes before Stage B is called."""
        store, ep_queue, cohort_queue, catalog, registry = _setup(tmp_path)
        _seed_registry(registry, store)

        episode_ids = _seed_episodes(store, ep_queue, catalog, n=5)
        rob_model = MockRobometerModel(seed=42)

        # High threshold — most should fail
        run_stage_a(
            model=rob_model, queue=ep_queue, store=store, catalog=catalog,
            robometer_threshold=0.99,  # nearly all fail
            poll_wait=0, max_episodes=5,
        )

        passed = [ep_id for ep_id in episode_ids
                  if catalog.get_row(ep_id).robometer_pass is True]
        failed = [ep_id for ep_id in episode_ids
                  if catalog.get_row(ep_id).robometer_pass is False]

        # Emit cohort with only passed episodes
        if passed:
            cohort_id = _emit_cohort(catalog, cohort_queue, episode_ids)
            dem_model = MockDeminfModel(seed=42)
            run_stage_b(
                model=dem_model, queue=cohort_queue, store=store, catalog=catalog,
                registry=registry, deminf_threshold=-999.0,
                poll_wait=0, max_cohorts=1,
            )

        # Failed episodes should not have deminf scores
        for ep_id in failed:
            row = catalog.get_row(ep_id)
            assert row.deminf_score is None


class TestE2EParallel:
    """Parallel mode: Stage B fires when min episodes accumulate."""

    def test_pipeline_parallel(self, tmp_path):
        from rdf.stage_b_deminf.accumulator import accumulate_sequential

        store, ep_queue, cohort_queue, catalog, registry = _setup(tmp_path)
        _seed_registry(registry, store)
        episode_ids = _seed_episodes(store, ep_queue, catalog, n=6)

        rob_model = MockRobometerModel(seed=42)
        run_stage_a(
            model=rob_model, queue=ep_queue, store=store, catalog=catalog,
            robometer_threshold=0.0, poll_wait=0, max_episodes=6,
        )

        cfg = PipelineConfig(mode="parallel", deminf_cohort_min=3)
        cohort_ids = accumulate_sequential(
            tasks=[TASK], catalog=catalog, cohort_queue=cohort_queue,
            pipeline_cfg=cfg, vae_version="v1", reference_set_version="v1",
        )
        assert len(cohort_ids) == 1

        dem_model = MockDeminfModel(seed=42)
        run_stage_b(
            model=dem_model, queue=cohort_queue, store=store, catalog=catalog,
            registry=registry, deminf_threshold=-999.0,
            poll_wait=0, max_cohorts=1,
        )

        counts = decide_all(TASK, catalog, 0.0, -999.0)
        assert counts["pending"] == 0

    def test_no_duplicates_on_restart(self, tmp_path):
        """Restarting Stage B with same episodes produces no duplicate scores."""
        store, ep_queue, cohort_queue, catalog, registry = _setup(tmp_path)
        _seed_registry(registry, store)
        episode_ids = _seed_episodes(store, ep_queue, catalog, n=4)

        rob_model = MockRobometerModel(seed=42)
        run_stage_a(
            model=rob_model, queue=ep_queue, store=store, catalog=catalog,
            robometer_threshold=0.0, poll_wait=0, max_episodes=4,
        )

        cohort_id = _emit_cohort(catalog, cohort_queue, episode_ids)
        dem_model = MockDeminfModel(seed=42)

        # First run
        run_stage_b(
            model=dem_model, queue=cohort_queue, store=store, catalog=catalog,
            registry=registry, deminf_threshold=-999.0, poll_wait=0, max_cohorts=1,
        )

        first_scores = {ep: catalog.get_row(ep).deminf_score for ep in episode_ids}

        # Re-emit same cohort and run again
        _emit_cohort(catalog, cohort_queue, episode_ids)
        run_stage_b(
            model=dem_model, queue=cohort_queue, store=store, catalog=catalog,
            registry=registry, deminf_threshold=-999.0, poll_wait=0, max_cohorts=1,
        )

        # Scores should be identical (idempotency)
        for ep_id in episode_ids:
            assert catalog.get_row(ep_id).deminf_score == first_scores[ep_id]
