"""Frozen Pydantic v2 schemas — all components read/write these."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class EpisodeManifest(BaseModel):
    model_config = {"frozen": True}

    episode_id: str
    s3_prefix: str
    robot_id: str
    embodiment: str
    task: str
    instruction: str
    head_video_key: str
    mcap_key: str
    metadata_key: str
    created_at: datetime
    schema_version: str = "1"


class RobometerResult(BaseModel):
    model_config = {"frozen": True}

    episode_id: str
    task: str
    embodiment: str
    robometer_reward: float
    robometer_success_pred: float
    frames_used: int
    model_version: str
    latency_ms: float
    scored_at: datetime
    status: Literal["scored", "failed"]
    error: str | None = None


class CohortMessage(BaseModel):
    model_config = {"frozen": True}

    cohort_id: str
    task: str
    episode_ids: list[str]
    vae_version: str
    reference_set_version: str
    created_at: datetime


class DeminfResult(BaseModel):
    model_config = {"frozen": True}

    episode_id: str
    task: str
    deminf_score: float
    vae_version: str
    reference_set_version: str
    cohort_id: str
    scored_at: datetime
    status: Literal["scored", "failed"]
    error: str | None = None


class CatalogRow(BaseModel):
    episode_id: str
    task: str
    embodiment: str
    robot_id: str
    robometer_reward: float | None = None
    robometer_success_pred: float | None = None
    robometer_pass: bool | None = None
    deminf_score: float | None = None
    deminf_pass: bool | None = None
    final_decision: Literal["keep", "drop", "pending"] = "pending"
    reasons: list[str] = Field(default_factory=list)
    robometer_model_version: str | None = None
    vae_version: str | None = None
    pipeline_mode: Literal["sequential", "parallel"]
    created_at: datetime
    updated_at: datetime


# ── Config models ──────────────────────────────────────────────────────────────


class EmbodimentConfig(BaseModel):
    model_config = {"frozen": True}

    name: str
    head_camera: str
    instruction_field: str
    # DECISION-NEEDED: real mcap topic names — synthetic defaults below
    state_topics: list[str] = Field(default_factory=lambda: ["/obs/rgb"])
    action_topics: list[str] = Field(default_factory=lambda: ["/action"])
    action_chunk_size: int = 1


class ThresholdConfig(BaseModel):
    model_config = {"frozen": True}

    task: str
    robometer_threshold: float
    deminf_threshold: float
    calibrated_against: str
    vae_version: str


class PipelineConfig(BaseModel):
    model_config = {"frozen": True}

    mode: Literal["sequential", "parallel"] = "sequential"
    raw_bucket: str = "rdf-raw"
    clean_bucket: str = "rdf-clean"
    registry_bucket: str = "rdf-registry"
    episode_queue_name: str = "rdf-episodes"
    cohort_queue_name: str = "rdf-cohorts"
    deminf_cohort_min: int = 50
    deminf_cohort_max: int = 300
    deminf_cohort_timeout_hours: float = 12.0
