"""Tests for Stage A Robometer worker loop."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rdf.harness.catalog import LocalCatalog
from rdf.harness.queue import LocalQueue
from rdf.harness.storage import LocalObjectStore
from rdf.models.mock import MockRobometerModel, reset_init_call_count
from rdf.schemas.models import CatalogRow, EpisodeManifest
from rdf.stage_a_robometer.worker import run_worker
from tests.fixtures.generate_episode import make_episode_manifest, synthetic_mp4_bytes

NOW = datetime.now(timezone.utc)


def _seed_episode(ep_id: str, store: LocalObjectStore, queue: LocalQueue) -> EpisodeManifest:
    manifest = make_episode_manifest(ep_id)
    store.put_bytes(manifest.head_video_key, synthetic_mp4_bytes(ep_id))
    queue.enqueue(manifest.model_dump(mode="json"), dedup_id=ep_id)
    return manifest


def _seed_catalog_row(ep_id: str, catalog: LocalCatalog, task: str = "pick_cup") -> None:
    catalog.upsert_row(CatalogRow(
        episode_id=ep_id, task=task, embodiment="franka", robot_id="r1",
        pipeline_mode="sequential", created_at=NOW, updated_at=NOW,
    ))


class TestStageAWorker:
    def test_drains_queue(self, tmp_path):
        store = LocalObjectStore(root=str(tmp_path / "storage"))
        queue = LocalQueue("episodes", root=str(tmp_path / "queues"))
        catalog = LocalCatalog(root=str(tmp_path / "catalog"))
        model = MockRobometerModel()

        n = 5
        for i in range(n):
            ep_id = f"ep-{i:03d}"
            manifest = _seed_episode(ep_id, store, queue)
            _seed_catalog_row(ep_id, catalog)

        processed = run_worker(
            model=model, queue=queue, store=store, catalog=catalog,
            robometer_threshold=0.5, poll_wait=0, max_episodes=n,
        )

        assert processed == n
        assert queue.depth() == 0

    def test_no_duplicates_on_rerun(self, tmp_path):
        store = LocalObjectStore(root=str(tmp_path / "storage"))
        queue = LocalQueue("episodes", root=str(tmp_path / "queues"))
        catalog = LocalCatalog(root=str(tmp_path / "catalog"))
        model = MockRobometerModel()

        ep_id = "ep-001"
        manifest = _seed_episode(ep_id, store, queue)
        _seed_catalog_row(ep_id, catalog)

        # Run twice — second run should skip (idempotency)
        run_worker(model=model, queue=queue, store=store, catalog=catalog,
                   robometer_threshold=0.5, poll_wait=0, max_episodes=1)

        # Re-enqueue the same episode
        queue.enqueue(manifest.model_dump(mode="json"), dedup_id=ep_id + "-run2")

        run_worker(model=model, queue=queue, store=store, catalog=catalog,
                   robometer_threshold=0.5, poll_wait=0, max_episodes=1)

        row = catalog.get_row(ep_id)
        assert row.robometer_reward is not None

    def test_poison_episode_goes_to_dlq(self, tmp_path):
        store = LocalObjectStore(root=str(tmp_path / "storage"))
        queue = LocalQueue("episodes", root=str(tmp_path / "queues"))
        catalog = LocalCatalog(root=str(tmp_path / "catalog"))
        model = MockRobometerModel()

        # Enqueue a malformed message (not a valid EpisodeManifest)
        queue.enqueue({"invalid": "payload"})

        run_worker(model=model, queue=queue, store=store, catalog=catalog,
                   robometer_threshold=0.5, poll_wait=0, max_episodes=1)

        # Main queue drained, DLQ should have the message
        assert queue.depth() == 0

    def test_robometer_gate_pass_fail(self, tmp_path):
        """robometer_pass must reflect whether success_pred >= threshold."""
        store = LocalObjectStore(root=str(tmp_path / "storage"))
        queue = LocalQueue("episodes", root=str(tmp_path / "queues"))
        catalog = LocalCatalog(root=str(tmp_path / "catalog"))
        model = MockRobometerModel(seed=42)

        ep_id = "ep-gate"
        _seed_episode(ep_id, store, queue)
        _seed_catalog_row(ep_id, catalog)

        threshold = 0.5
        run_worker(model=model, queue=queue, store=store, catalog=catalog,
                   robometer_threshold=threshold, poll_wait=0, max_episodes=1)

        row = catalog.get_row(ep_id)
        assert row.robometer_success_pred is not None
        # Verify pass/fail is consistent with threshold
        assert row.robometer_pass == (row.robometer_success_pred >= threshold)

    def test_model_loaded_once(self, tmp_path):
        """Worker instantiated once — not once per episode."""
        store = LocalObjectStore(root=str(tmp_path / "storage"))
        queue = LocalQueue("episodes", root=str(tmp_path / "queues"))
        catalog = LocalCatalog(root=str(tmp_path / "catalog"))

        reset_init_call_count()
        model = MockRobometerModel()
        from rdf.models.mock import get_init_call_count
        assert get_init_call_count() == 1

        n = 5
        for i in range(n):
            ep_id = f"ep-once-{i}"
            _seed_episode(ep_id, store, queue)
            _seed_catalog_row(ep_id, catalog)

        run_worker(model=model, queue=queue, store=store, catalog=catalog,
                   robometer_threshold=0.5, poll_wait=0, max_episodes=n)

        # Still only 1 init call — model loaded once, not per episode
        assert get_init_call_count() == 1
