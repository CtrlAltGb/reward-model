"""Tests for harness backends (local implementations)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from rdf.harness.catalog import LocalCatalog
from rdf.harness.queue import LocalQueue
from rdf.harness.storage import LocalObjectStore
from rdf.schemas.models import CatalogRow, DeminfResult, RobometerResult

NOW = datetime.now(timezone.utc)


def _make_catalog_row(episode_id: str, task: str = "pick_cup") -> CatalogRow:
    return CatalogRow(
        episode_id=episode_id, task=task, embodiment="franka", robot_id="r1",
        pipeline_mode="sequential", created_at=NOW, updated_at=NOW,
    )


class TestLocalObjectStore:
    def test_put_get_roundtrip(self, local_store):
        local_store.put_bytes("foo/bar.txt", b"hello")
        assert local_store.get_bytes("foo/bar.txt") == b"hello"

    def test_exists(self, local_store):
        assert not local_store.exists("missing.txt")
        local_store.put_bytes("present.txt", b"x")
        assert local_store.exists("present.txt")

    def test_delete(self, local_store):
        local_store.put_bytes("del.txt", b"data")
        local_store.delete("del.txt")
        assert not local_store.exists("del.txt")

    def test_list_keys(self, local_store):
        local_store.put_bytes("prefix/a.txt", b"1")
        local_store.put_bytes("prefix/b.txt", b"2")
        local_store.put_bytes("other/c.txt", b"3")
        keys = local_store.list_keys("prefix/")
        assert len(keys) == 2
        assert all("prefix" in k for k in keys)


class TestLocalQueue:
    def test_enqueue_receive_delete(self, episode_queue):
        episode_queue.enqueue({"ep": "1"})
        msgs = episode_queue.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body == {"ep": "1"}
        episode_queue.delete(msgs[0].receipt)
        assert episode_queue.depth() == 0

    def test_dedup_id(self, episode_queue):
        episode_queue.enqueue({"ep": "1"}, dedup_id="ep-001")
        episode_queue.enqueue({"ep": "1-dup"}, dedup_id="ep-001")
        assert episode_queue.depth() == 1

    def test_dlq(self, episode_queue):
        episode_queue.send_to_dlq({"ep": "poison"})
        msgs = episode_queue.receive()
        assert len(msgs) == 0

    def test_depth(self, episode_queue):
        episode_queue.enqueue({"a": 1})
        episode_queue.enqueue({"b": 2})
        assert episode_queue.depth() == 2


class TestLocalCatalog:
    def test_upsert_get(self, local_catalog):
        row = _make_catalog_row("ep-001")
        local_catalog.upsert_row(row)
        got = local_catalog.get_row("ep-001")
        assert got is not None
        assert got.episode_id == "ep-001"

    def test_get_missing(self, local_catalog):
        assert local_catalog.get_row("nonexistent") is None

    def test_update_robometer(self, local_catalog):
        local_catalog.upsert_row(_make_catalog_row("ep-001"))
        result = RobometerResult(
            episode_id="ep-001", task="pick_cup", embodiment="franka",
            robometer_reward=0.7, robometer_success_pred=0.8, frames_used=8,
            model_version="v1", latency_ms=100.0, scored_at=NOW, status="scored",
        )
        local_catalog.update_robometer(result, pass_=True, threshold=0.5)
        row = local_catalog.get_row("ep-001")
        assert row.robometer_reward == 0.7
        assert row.robometer_pass is True

    def test_update_deminf(self, local_catalog):
        local_catalog.upsert_row(_make_catalog_row("ep-001"))
        result = DeminfResult(
            episode_id="ep-001", task="pick_cup", deminf_score=0.6,
            vae_version="v1", reference_set_version="v1", cohort_id="c1",
            scored_at=NOW, status="scored",
        )
        local_catalog.update_deminf(result, pass_=True, threshold=0.0)
        row = local_catalog.get_row("ep-001")
        assert row.deminf_score == 0.6

    def test_finalize(self, local_catalog):
        local_catalog.upsert_row(_make_catalog_row("ep-001"))
        local_catalog.finalize("ep-001", "keep", [])
        row = local_catalog.get_row("ep-001")
        assert row.final_decision == "keep"

    def test_rows_for_task(self, local_catalog):
        local_catalog.upsert_row(_make_catalog_row("ep-001", task="task_a"))
        local_catalog.upsert_row(_make_catalog_row("ep-002", task="task_a"))
        local_catalog.upsert_row(_make_catalog_row("ep-003", task="task_b"))
        rows = local_catalog.rows_for_task("task_a")
        assert len(rows) == 2
