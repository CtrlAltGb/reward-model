"""Generate synthetic episode fixtures for testing."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import yaml

from rdf.schemas.models import EpisodeManifest


def synthetic_mp4_bytes(episode_id: str) -> bytes:
    """Return fake MP4 bytes deterministic for the episode_id."""
    seed = int.from_bytes(hashlib.md5(episode_id.encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    return rng.bytes(1024)


def synthetic_mcap_bytes(episode_id: str) -> bytes:
    seed = int.from_bytes(hashlib.md5((episode_id + "mcap").encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    return rng.bytes(640)


def synthetic_metadata(
    episode_id: str,
    task: str = "pick_cup",
    robot_id: str = "r1",
    embodiment: str = "franka",
    instruction: str = "Pick up the cup",
) -> bytes:
    meta = {
        "robot_id": robot_id,
        "embodiment": embodiment,
        "task": task,
        "instruction": instruction,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return yaml.safe_dump(meta).encode()


def make_episode_manifest(
    episode_id: str,
    task: str = "pick_cup",
    robot_id: str = "r1",
    embodiment: str = "franka",
    instruction: str = "Pick up the cup",
) -> EpisodeManifest:
    return EpisodeManifest(
        episode_id=episode_id,
        s3_prefix=f"raw/{episode_id}/",
        robot_id=robot_id,
        embodiment=embodiment,
        task=task,
        instruction=instruction,
        head_video_key=f"raw/{episode_id}/head.mp4",
        mcap_key=f"raw/{episode_id}/data.mcap",
        metadata_key=f"raw/{episode_id}/metadata.yaml",
        created_at=datetime.now(timezone.utc),
    )
