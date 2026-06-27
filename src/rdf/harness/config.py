"""Config loading — reads configs/* YAML files, validates with Pydantic."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from rdf.schemas.models import (
    EmbodimentConfig,
    ModelsConfig,
    PathsConfig,
    PipelineConfig,
    ThresholdConfig,
)

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
    base = PipelineConfig.model_validate(_load_yaml(path)) if path.exists() else PipelineConfig()
    overrides: dict[str, Any] = {}
    if v := os.environ.get("RDF_ROBOMETER_THRESHOLD"):
        overrides["robometer_threshold"] = float(v)
    if v := os.environ.get("RDF_DEMINF_THRESHOLD"):
        overrides["deminf_threshold"] = float(v)
    if not overrides:
        return base
    return PipelineConfig.model_validate({**base.model_dump(), **overrides})


@lru_cache(maxsize=None)
def get_paths_config() -> PathsConfig:
    path = _configs_dir() / "paths.yaml"
    base = PathsConfig.model_validate(_load_yaml(path)) if path.exists() else PathsConfig()
    overrides: dict[str, Any] = {}
    if v := os.environ.get("RDF_DEMINF_DATA"):
        overrides["deminf_data_dir"] = v
    if v := os.environ.get("RDF_DEMINF_SCORES"):
        overrides["deminf_scores_file"] = v
    if not overrides:
        return base
    return PathsConfig.model_validate({**base.model_dump(), **overrides})


@lru_cache(maxsize=None)
def get_models_config() -> ModelsConfig:
    path = _configs_dir() / "models.yaml"
    base = ModelsConfig.model_validate(_load_yaml(path)) if path.exists() else ModelsConfig()
    overrides: dict[str, Any] = {}
    if v := os.environ.get("RDF_ROBOMETER_MODEL_VERSION"):
        overrides["robometer_model_version"] = v
    if v := os.environ.get("RDF_DEMINF_SPLIT"):
        overrides["deminf_split"] = v
    if not overrides:
        return base
    return ModelsConfig.model_validate({**base.model_dump(), **overrides})


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
    get_paths_config.cache_clear()
    get_models_config.cache_clear()
    get_embodiment_config.cache_clear()
    get_threshold_config.cache_clear()
