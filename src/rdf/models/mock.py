"""Deterministic, seedable mock implementations. No torch/jax imports.

Used in all tests via RDF_MODELS=mock (the default).
"""

from __future__ import annotations

import hashlib

import numpy as np

from rdf.models.base import DeminfModel, RobometerModel, RobometerScore

_MOCK_MODEL_VERSION = "mock-v1"
_MOCK_VAE_VERSION = "mock-vae-v1"

_init_call_count = 0


def get_init_call_count() -> int:
    return _init_call_count


def reset_init_call_count() -> None:
    global _init_call_count
    _init_call_count = 0


class MockRobometerModel:
    """Deterministic mock: scores derived from hash of (video_path, instruction).

    Tracks __init__ calls so tests can assert model loads exactly once.
    """

    def __init__(self, seed: int = 42):
        global _init_call_count
        _init_call_count += 1
        self.seed = seed
        self.model_version = _MOCK_MODEL_VERSION

    def score_episode(self, video_path: str, instruction: str) -> RobometerScore:
        digest = hashlib.md5(f"{video_path}:{instruction}:{self.seed}".encode()).digest()
        reward = (digest[0] / 255.0) * 0.9 + 0.05
        success_pred = (digest[1] / 255.0) * 0.9 + 0.05
        return RobometerScore(
            reward=round(reward, 4),
            success_pred=round(success_pred, 4),
            frames_used=8,
            model_version=self.model_version,
        )

    def score_episode_from_frames(self, frames: np.ndarray, instruction: str) -> RobometerScore:
        """Accept preprocessed frames (T, H, W, C) directly."""
        digest = hashlib.md5(f"frames:{frames.shape}:{instruction}:{self.seed}".encode()).digest()
        reward = (digest[0] / 255.0) * 0.9 + 0.05
        success_pred = (digest[1] / 255.0) * 0.9 + 0.05
        return RobometerScore(
            reward=round(reward, 4),
            success_pred=round(success_pred, 4),
            frames_used=int(frames.shape[0]),
            model_version=self.model_version,
        )


class MockDeminfModel:
    """Deterministic mock: encodes to fixed-dim latents, scores via dot product."""

    LATENT_DIM = 16

    def __init__(self, seed: int = 42):
        global _init_call_count
        _init_call_count += 1
        self.seed = seed
        self.vae_version = _MOCK_VAE_VERSION
        self._rng = np.random.default_rng(seed)

    def encode_episodes(
        self,
        episode_states: list[np.ndarray],
        episode_actions: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(episode_states)
        rng = np.random.default_rng(self.seed + 1)
        obs_latents = rng.standard_normal((n, self.LATENT_DIM)).astype(np.float32)
        action_latents = rng.standard_normal((n, self.LATENT_DIM)).astype(np.float32)
        return obs_latents, action_latents

    def score_against_reference(
        self,
        obs_latents: np.ndarray,
        action_latents: np.ndarray,
        reference_obs_latents: np.ndarray,
        reference_action_latents: np.ndarray,
        k: int = 5,
    ) -> list[float]:
        n = len(obs_latents)
        # Simple mock: score = 1 - normalized L2 distance to reference mean
        ref_obs_mean = reference_obs_latents.mean(axis=0)
        ref_act_mean = reference_action_latents.mean(axis=0)
        scores = []
        for i in range(n):
            dist = float(
                np.linalg.norm(obs_latents[i] - ref_obs_mean)
                + np.linalg.norm(action_latents[i] - ref_act_mean)
            )
            score = max(0.0, 1.0 - dist / (2 * self.LATENT_DIM**0.5))
            scores.append(round(score, 4))
        return scores


def get_robometer_model(**kwargs) -> RobometerModel:
    return MockRobometerModel(**kwargs)


def get_deminf_model(**kwargs) -> DeminfModel:
    return MockDeminfModel(**kwargs)
