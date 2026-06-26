"""Shared test fixtures."""

from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault("RDF_MODELS", "mock")
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


@pytest.fixture
def mock_robometer():
    from rdf.models.mock import MockRobometerModel, reset_init_call_count
    reset_init_call_count()
    return MockRobometerModel(seed=42)


@pytest.fixture
def mock_deminf():
    from rdf.models.mock import MockDeminfModel, reset_init_call_count
    reset_init_call_count()
    return MockDeminfModel(seed=42)
