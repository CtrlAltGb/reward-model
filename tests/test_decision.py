"""Tests for decision, registry, and materializer."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from rdf.decision.decide import decide_all, decide_episode
from rdf.decision.materialize import materialize_task
from rdf.harness.catalog import LocalCatalog
from rdf.harness.storage import LocalObjectStore
from rdf.models.registry import VaeArtifact, VaeRegistry
from rdf.schemas.models import CatalogRow

NOW = datetime.now(timezone.utc)


def _make_row(
    ep_id: str,
    rob_success: float | None = None,
    deminf: float | None = None,
    task: str = "pick_cup",
) -> CatalogRow:
    return CatalogRow(
        episode_id=ep_id, task=task, embodiment="franka", robot_id="r1",
        robometer_reward=rob_success,
        robometer_success_pred=rob_success,
        robometer_pass=(rob_success is not None and rob_success >= 0.5) if rob_success is not None else None,
        deminf_score=deminf,
        deminf_pass=(deminf is not None and deminf >= 0.0) if deminf is not None else None,
        robometer_model_version="v1" if rob_success is not None else None,
        vae_version="v1" if deminf is not None else None,
        pipeline_mode="sequential",
        created_at=NOW, updated_at=NOW,
    )


class TestDecide:
    def test_keep(self, local_catalog):
        row = _make_row("ep-1", rob_success=0.9, deminf=0.5)
        local_catalog.upsert_row(row)
        decision = decide_episode("ep-1", local_catalog, 0.5, 0.0)
        assert decision == "keep"

    def test_drop_robometer_gate(self, local_catalog):
        row = _make_row("ep-1", rob_success=0.1, deminf=0.5)
        local_catalog.upsert_row(row)
        decision = decide_episode("ep-1", local_catalog, 0.5, 0.0)
        assert decision == "drop"
        assert "task_incomplete" in local_catalog.get_row("ep-1").reasons

    def test_drop_deminf_trim(self, local_catalog):
        row = _make_row("ep-1", rob_success=0.9, deminf=-1.0)
        local_catalog.upsert_row(row)
        decision = decide_episode("ep-1", local_catalog, 0.5, 0.0)
        assert decision == "drop"
        assert "low_quality_jitter" in local_catalog.get_row("ep-1").reasons

    def test_pending_when_not_scored(self, local_catalog):
        row = _make_row("ep-1", rob_success=0.9, deminf=None)
        local_catalog.upsert_row(row)
        decision = decide_episode("ep-1", local_catalog, 0.5, 0.0)
        assert decision == "pending"

    def test_robometer_gate_before_deminf(self, local_catalog):
        """Robometer gate must fire before DemInf is checked."""
        row = _make_row("ep-1", rob_success=0.1, deminf=None)
        local_catalog.upsert_row(row)
        decision = decide_episode("ep-1", local_catalog, 0.5, 0.0)
        assert decision == "drop"
        assert "task_incomplete" in local_catalog.get_row("ep-1").reasons

    def test_decide_all(self, local_catalog):
        for i, (rob, dem) in enumerate([(0.9, 0.5), (0.1, 0.5), (0.9, -1.0)]):
            local_catalog.upsert_row(_make_row(f"ep-{i}", rob, dem))
        counts = decide_all("pick_cup", local_catalog, 0.5, 0.0)
        assert counts["keep"] == 1
        assert counts["drop"] == 2


class TestVaeRegistry:
    def test_publish_load_roundtrip(self, local_store):
        registry = VaeRegistry(store=local_store)
        obs_latents = np.random.random((10, 16)).astype(np.float32)
        act_latents = np.random.random((10, 16)).astype(np.float32)
        artifact = VaeArtifact(
            task="pick_cup", vae_version="v1",
            obs_ckpt="/tmp/obs", action_ckpt="/tmp/act",
            reference_latents_obs=obs_latents,
            reference_latents_action=act_latents,
        )
        registry.publish(artifact)
        loaded = registry.load("pick_cup", "v1")
        np.testing.assert_array_almost_equal(loaded.reference_latents_obs, obs_latents)
        np.testing.assert_array_almost_equal(loaded.reference_latents_action, act_latents)
        assert loaded.obs_ckpt == "/tmp/obs"

    def test_current_version(self, local_store):
        registry = VaeRegistry(store=local_store)
        for v in ["v1", "v2", "v10"]:
            obs = np.zeros((5, 16), dtype=np.float32)
            artifact = VaeArtifact("pick_cup", v, "/tmp/obs", None, obs, obs)
            registry.publish(artifact)
        assert registry.current_version("pick_cup") == "v10"


class TestMaterialize:
    def test_copies_kept_episodes(self, tmp_path, local_catalog):
        raw_store = LocalObjectStore(root=str(tmp_path / "raw"))
        clean_store = LocalObjectStore(root=str(tmp_path / "clean"))

        for ep_id in ["ep-keep", "ep-drop"]:
            for fname in ["head.mp4", "data.mcap", "metadata.yaml"]:
                raw_store.put_bytes(f"raw/{ep_id}/{fname}", b"data")

        local_catalog.upsert_row(CatalogRow(
            episode_id="ep-keep", task="pick_cup", embodiment="franka", robot_id="r1",
            final_decision="keep", pipeline_mode="sequential", created_at=NOW, updated_at=NOW,
        ))
        local_catalog.upsert_row(CatalogRow(
            episode_id="ep-drop", task="pick_cup", embodiment="franka", robot_id="r1",
            final_decision="drop", pipeline_mode="sequential", created_at=NOW, updated_at=NOW,
        ))

        counts = materialize_task("pick_cup", local_catalog, raw_store, clean_store)
        assert counts["copied"] == 3  # 3 files for ep-keep
        assert not clean_store.exists("clean/task=pick_cup/episode=ep-drop/head.mp4")
        assert clean_store.exists("clean/task=pick_cup/episode=ep-keep/head.mp4")

    def test_idempotent(self, tmp_path, local_catalog):
        raw_store = LocalObjectStore(root=str(tmp_path / "raw"))
        clean_store = LocalObjectStore(root=str(tmp_path / "clean"))

        for fname in ["head.mp4", "data.mcap", "metadata.yaml"]:
            raw_store.put_bytes(f"raw/ep-keep/{fname}", b"data")

        local_catalog.upsert_row(CatalogRow(
            episode_id="ep-keep", task="pick_cup", embodiment="franka", robot_id="r1",
            final_decision="keep", pipeline_mode="sequential", created_at=NOW, updated_at=NOW,
        ))

        materialize_task("pick_cup", local_catalog, raw_store, clean_store)
        c2 = materialize_task("pick_cup", local_catalog, raw_store, clean_store)
        assert c2["copied"] == 0
        assert c2["skipped"] == 3
