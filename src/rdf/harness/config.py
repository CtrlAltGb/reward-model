"""Config loading — reads configs/* YAML files, validates with Pydantic."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from rdf.schemas.models import EmbodimentConfig, PipelineConfig, ThresholdConfig

_CONFIGS_DIR = Path(__file__).parent.parent.parent.parent / "configs"


def _configs_dir() -> Path:
    override = os.environ.get("RDF_CONFIGS_DIR")
    return Path(override) if override else _CONFIGS_DIR


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=None)
def get_pipeline_config() -> PipelineConfig:
    path = _configs_dir() / "pipeline.yaml"
    if path.exists():
        return PipelineConfig.model_validate(_load_yaml(path))
    return PipelineConfig()


@lru_cache(maxsize=None)
def get_embodiment_config(name: str) -> EmbodimentConfig:
    path = _configs_dir() / "embodiments" / f"{name}.yaml"
    if path.exists():
        return EmbodimentConfig.model_validate(_load_yaml(path))
    return EmbodimentConfig(name=name, head_camera="head", instruction_field="instruction")


@lru_cache(maxsize=None)
def get_threshold_config(task: str) -> ThresholdConfig | None:
    path = _configs_dir() / "thresholds" / f"{task}.yaml"
    if path.exists():
        return ThresholdConfig.model_validate(_load_yaml(path))
    return None


def clear_config_cache() -> None:
    get_pipeline_config.cache_clear()
    get_embodiment_config.cache_clear()
    get_threshold_config.cache_clear()
