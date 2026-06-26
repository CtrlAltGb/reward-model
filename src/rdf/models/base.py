"""Protocol interfaces for model workers — loaded once, called many times."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class RobometerScore:
    reward: float
    success_pred: float
    frames_used: int
    model_version: str


class RobometerModel(Protocol):
    """Loaded once at startup, called per episode."""

    def score_episode(self, video_path: str, instruction: str) -> RobometerScore: ...


class DeminfModel(Protocol):
    """Loaded once at startup, called per cohort."""

    def encode_episodes(
        self,
        episode_states: list[np.ndarray],
        episode_actions: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode episodes → (obs_latents, action_latents), each (N, latent_dim)."""
        ...

    def score_against_reference(
        self,
        obs_latents: np.ndarray,
        action_latents: np.ndarray,
        reference_obs_latents: np.ndarray,
        reference_action_latents: np.ndarray,
        k: int = 5,
    ) -> list[float]:
        """kNN-MI score per episode — higher = better quality."""
        ...
