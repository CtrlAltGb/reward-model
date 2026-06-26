"""Test schema round-trips to/from JSON and YAML."""

from __future__ import annotations

from datetime import datetime, timezone

import yaml

from rdf.schemas.models import (
    CatalogRow,
    CohortMessage,
    DeminfResult,
    EmbodimentConfig,
    EpisodeManifest,
    PipelineConfig,
    RobometerResult,
    ThresholdConfig,
)

NOW = datetime.now(timezone.utc)


def test_episode_manifest_roundtrip():
    ep = EpisodeManifest(
        episode_id="ep-001", s3_prefix="raw/ep-001/", robot_id="r1",
        embodiment="franka", task="pick_cup", instruction="Pick up the cup",
        head_video_key="raw/ep-001/head.mp4", mcap_key="raw/ep-001/data.mcap",
        metadata_key="raw/ep-001/metadata.yaml", created_at=NOW,
    )
    assert EpisodeManifest.model_validate_json(ep.model_dump_json()) == ep
    assert EpisodeManifest.model_validate(yaml.safe_load(yaml.safe_dump(ep.model_dump()))) is not None


def test_robometer_result_roundtrip():
    rr = RobometerResult(
        episode_id="ep-001", task="pick_cup", embodiment="franka",
        robometer_reward=0.72, robometer_success_pred=0.85, frames_used=8,
        model_version="v1", latency_ms=120.0, scored_at=NOW, status="scored",
    )
    assert RobometerResult.model_validate_json(rr.model_dump_json()) == rr


def test_cohort_message_roundtrip():
    coh = CohortMessage(
        cohort_id="coh-001", task="pick_cup", episode_ids=["ep-001", "ep-002"],
        vae_version="v1", reference_set_version="v1", created_at=NOW,
    )
    assert CohortMessage.model_validate_json(coh.model_dump_json()) == coh


def test_deminf_result_roundtrip():
    dr = DeminfResult(
        episode_id="ep-001", task="pick_cup", deminf_score=0.65,
        vae_version="v1", reference_set_version="v1", cohort_id="coh-001",
        scored_at=NOW, status="scored",
    )
    assert DeminfResult.model_validate_json(dr.model_dump_json()) == dr


def test_catalog_row_roundtrip():
    cat = CatalogRow(
        episode_id="ep-001", task="pick_cup", embodiment="franka", robot_id="r1",
        final_decision="pending", pipeline_mode="sequential",
        created_at=NOW, updated_at=NOW,
    )
    assert CatalogRow.model_validate_json(cat.model_dump_json()) == cat


def test_catalog_row_with_scores():
    cat = CatalogRow(
        episode_id="ep-001", task="pick_cup", embodiment="franka", robot_id="r1",
        robometer_reward=0.7, robometer_success_pred=0.8, robometer_pass=True,
        deminf_score=0.6, deminf_pass=True, final_decision="keep",
        reasons=[], robometer_model_version="v1", vae_version="vae-v1",
        pipeline_mode="sequential", created_at=NOW, updated_at=NOW,
    )
    assert CatalogRow.model_validate_json(cat.model_dump_json()) == cat


def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.mode == "sequential"
    assert cfg.deminf_cohort_min == 50
    assert cfg.deminf_cohort_max == 300


def test_embodiment_config():
    ec = EmbodimentConfig(name="franka", head_camera="head", instruction_field="instruction")
    assert ec.state_topics == ["/obs/rgb"]
    assert ec.action_topics == ["/action"]
