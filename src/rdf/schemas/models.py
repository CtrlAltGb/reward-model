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
    task_id: str
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
    robometer_progress_dip: bool | None = None
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
    # Decision thresholds
    robometer_threshold: float = 0.5
    deminf_threshold: float = -10.0
    # Monotonicity dip tolerance — max allowed per-frame decrease in progress.
    # E.g. 0.05 allows a drop of up to 5% between consecutive frames.
    # Set to 1.0 to disable the dip check entirely.
    robometer_progress_dip_tolerance: float = 0.05
    # If > 0, override deminf_threshold with the Nth percentile of scored episodes
    # (e.g. 0.20 drops the bottom 20%). Set to 0.0 to use fixed deminf_threshold.
    deminf_filter_bottom_pct: float = 0.0
    # Versioning
    vae_version: str = "v1"
    reference_set_version: str = "v1"
    # Default embodiment for Stage B
    embodiment: str = "franka"
    # Fallback task_id when metadata.yaml has no task_id field
    default_task_id: str = "001"
    # Default task instruction passed to Robometer when metadata.yaml has no instruction field.
    # Describes what a successful episode looks like so the VLM can score it correctly.
    default_instruction: str = ""


class PathsConfig(BaseModel):
    model_config = {"frozen": True}

    # Input data
    clean_data_dir: str = "/data/clean_data"
    # Pipeline scratch / outputs
    scratch_dir: str = "/data/reward_model_files/rdf_integration"
    deminf_data_dir: str = "/data/reward_model_files/rdf_pipeline_deminf/deminf_data"
    deminf_scores_file: str = "/data/reward_model_files/rdf_deminf_scores.json"
    deminf_ckpts_dir: str = "/data/reward_model_files/rdf_deminf_ckpts"
    # Local harness state dirs (SQLite catalog + queues)
    local_catalog_dir: str = "/data/reward_model_files/rdf_integration/catalog"
    local_queue_dir: str = "/data/reward_model_files/rdf_integration/queues"
    # Python environments
    openx_python: str = "/data/.conda/envs/openx/bin/python3"
    # Upstream repos — import-only, never modify
    robometer_root: str = "/data/robometer"
    deminf_root: str = "/data/demonstration-information"
    # Model checkpoints
    robometer_model_path: str = "/data/robometer/robometer/Robometer-4B"
    # HuggingFace cache root — set HF_HOME to this to keep all HF downloads in one place
    hf_home: str = "/data/.hf_cache"


class ModelsConfig(BaseModel):
    model_config = {"frozen": True}

    # Robometer — local worker
    robometer_model_version: str = "Robometer-4B"
    # Robometer — HTTP server (RobometerWorker only)
    robometer_server_url: str = "http://localhost:8001"
    robometer_server_timeout_s: float = 120.0
    robometer_server_fps: float = 1.0
    # Video preprocessing
    video_target_fps: float = 2.0
    video_target_size: int = 256
    video_preprocess_workers: int = 8
    decode_queue_maxsize: int = 8
    # DemInf scoring subprocess
    deminf_estimator: str = "ksg"
    deminf_batch_size: int = 32
    deminf_split: str = "train"
    # DemInf VAE training (auto-triggered when no checkpoint exists for a task_id)
    deminf_train_steps: int = 5000
    deminf_train_episodes: int = 50
    deminf_train_save_freq: int = 1000
