"""Extract state/action arrays from MCAP bytes.

Real MCAP topics are in configs/embodiments/<name>.yaml.
Synthetic reader for tests is also provided here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from rdf.schemas.models import EmbodimentConfig


class McapReader(Protocol):
    def read_topics(
        self, mcap_bytes: bytes, topics: list[str]
    ) -> dict[str, list[np.ndarray]]: ...


class RealMcapReader:
    """Uses the mcap library to read real MCAP files."""

    def read_topics(
        self, mcap_bytes: bytes, topics: list[str]
    ) -> dict[str, list[np.ndarray]]:
        import io

        from mcap.reader import make_reader

        result: dict[str, list[np.ndarray]] = {t: [] for t in topics}
        reader = make_reader(io.BytesIO(mcap_bytes))
        for schema, channel, message in reader.iter_messages(topics=topics):
            topic = channel.topic
            if topic in result:
                # DECISION-NEEDED: real deserialization depends on schema/encoding
                # For now, treat raw bytes as a flat float32 array
                arr = np.frombuffer(message.data, dtype=np.float32)
                result[topic].append(arr)
        return result


class SyntheticMcapReader:
    """Deterministic synthetic reader for tests — no real MCAP files needed."""

    def __init__(self, n_steps: int = 10, obs_dim: int = 64, action_dim: int = 7):
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def read_topics(
        self, mcap_bytes: bytes, topics: list[str]
    ) -> dict[str, list[np.ndarray]]:
        # Use the bytes hash as seed for determinism
        seed = int.from_bytes(mcap_bytes[:4], "little") if len(mcap_bytes) >= 4 else 42
        rng = np.random.default_rng(seed)
        result: dict[str, list[np.ndarray]] = {}
        for topic in topics:
            if "action" in topic:
                result[topic] = [rng.random(self.action_dim, dtype=np.float32) for _ in range(self.n_steps)]
            else:
                result[topic] = [rng.random(self.obs_dim, dtype=np.float32) for _ in range(self.n_steps)]
        return result


def extract_state_action(
    mcap_bytes: bytes,
    embodiment: EmbodimentConfig,
    reader: McapReader | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (states, actions) as (T, obs_dim) and (T, action_dim) arrays."""
    if reader is None:
        reader = RealMcapReader()
    all_topics = embodiment.state_topics + embodiment.action_topics
    data = reader.read_topics(mcap_bytes, all_topics)

    state_frames = []
    for topic in embodiment.state_topics:
        frames = data.get(topic, [])
        state_frames.append(np.stack(frames) if frames else np.zeros((0,), dtype=np.float32))

    action_frames = []
    for topic in embodiment.action_topics:
        frames = data.get(topic, [])
        action_frames.append(np.stack(frames) if frames else np.zeros((0,), dtype=np.float32))

    states = np.concatenate(state_frames, axis=-1) if state_frames else np.zeros((0,), dtype=np.float32)
    actions = np.concatenate(action_frames, axis=-1) if action_frames else np.zeros((0,), dtype=np.float32)
    return states, actions
