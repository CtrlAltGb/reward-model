"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("RDF_STORAGE", "local")
os.environ.setdefault("RDF_QUEUE", "local")
os.environ.setdefault("RDF_CATALOG", "local")


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def local_store(tmp_path):
    from rdf.harness.storage import LocalObjectStore
    return LocalObjectStore(root=str(tmp_path / "storage"))


@pytest.fixture
def episode_queue(tmp_path):
    from rdf.harness.queue import LocalQueue
    return LocalQueue(name="episodes", root=str(tmp_path / "queues"))


@pytest.fixture
def cohort_queue(tmp_path):
    from rdf.harness.queue import LocalQueue
    return LocalQueue(name="cohorts", root=str(tmp_path / "queues"))


@pytest.fixture
def local_catalog(tmp_path):
    from rdf.harness.catalog import LocalCatalog
    return LocalCatalog(root=str(tmp_path / "catalog"))
