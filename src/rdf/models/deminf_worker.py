"""DemInf persistent worker — REAL-MODEL SEAM.

Written by reading:
  /data/demonstration-information/scripts/quality/estimate_quality.py
  /data/demonstration-information/scripts/quality/quality_estimators.py
  /data/demonstration-information/openx/utils/evaluate.py  (load_checkpoint)

Loading pattern (from quality_estimators.get_dataset_and_score_fn / load_checkpoint):
  alg, state, _, _ = load_checkpoint(obs_ckpt)   # orbax CheckpointManager
  z = alg.predict(state, batch, rng)              # JAX array (B, latent_dim)

Scoring: kSG estimator (ksg_estimator from quality_estimators.py):
  I(obs;action) via kNN mutual information — higher = more consistent = better quality.

Must be imported/run inside the openx conda env:
  conda run -n openx python -m rdf.models.deminf_worker

DO NOT modify /data/demonstration-information — only import from it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from rdf.models.base import DeminfModel

_DEMINF_ROOT = Path("/data/demonstration-information")
if str(_DEMINF_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMINF_ROOT))
if str(_DEMINF_ROOT / "scripts" / "quality") not in sys.path:
    sys.path.insert(0, str(_DEMINF_ROOT / "scripts" / "quality"))

_VAE_VERSION = os.environ.get("RDF_DEMINF_VAE_VERSION", "v1")


def _l2_dists_np(z: np.ndarray) -> np.ndarray:
    """Pairwise L2 distances (B, B) — numpy fallback for small batches."""
    diff = z[:, None, :] - z[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def _knn_np(z: np.ndarray, ks: np.ndarray) -> np.ndarray:
    dist = _l2_dists_np(z)
    return np.sort(dist, axis=-1)[:, ks]


def _ksg_score_np(
    z_obs: np.ndarray,
    z_action: np.ndarray,
    ks: np.ndarray,
) -> np.ndarray:
    """numpy kSG MI estimator — used when JAX not available (testing)."""
    from scipy.special import digamma

    obs_dist = _l2_dists_np(z_obs)
    action_dist = _l2_dists_np(z_action)
    joint_dist = np.maximum(obs_dist, action_dist)
    joint_knn_dists = np.sort(joint_dist, axis=-1)[:, ks]
    obs_count = np.sum(obs_dist[:, :, None] < joint_knn_dists[:, None, :], axis=1)
    action_count = np.sum(action_dist[:, :, None] < joint_knn_dists[:, None, :], axis=1)
    return -np.mean(digamma(obs_count + 1e-6) + digamma(action_count + 1e-6), axis=-1)


class DeminfWorker:
    """Loads VAE checkpoints once; encodes and scores cohorts in a loop.

    The worker supports two modes:
      - JAX mode (default when running in openx env): uses alg.predict + jitted kSG
      - numpy fallback mode (testing): uses scipy kSG

    Reference latents are loaded from a pre-saved .npz file (produced by train_job.py).
    """

    def __init__(
        self,
        obs_ckpt: str | None = None,
        action_ckpt: str | None = None,
        reference_latents_path: str | None = None,
        vae_version: str | None = None,
        use_jax: bool = True,
    ):
        self.obs_ckpt = obs_ckpt or os.environ.get("RDF_DEMINF_OBS_CKPT", "")
        self.action_ckpt = action_ckpt or os.environ.get("RDF_DEMINF_ACTION_CKPT", "")
        self.reference_latents_path = reference_latents_path
        self.vae_version = vae_version or _VAE_VERSION
        self.use_jax = use_jax

        self._obs_alg = None
        self._obs_state = None
        self._action_alg = None
        self._action_state = None
        self._ref_obs_latents: np.ndarray | None = None
        self._ref_action_latents: np.ndarray | None = None

        if self.obs_ckpt:
            self._load_checkpoints()
        if self.reference_latents_path:
            self._load_reference_latents()

    def _load_checkpoints(self) -> None:
        from openx.utils.evaluate import load_checkpoint

        self._obs_alg, self._obs_state, _, _ = load_checkpoint(self.obs_ckpt)
        if self.action_ckpt:
            self._action_alg, self._action_state, _, _ = load_checkpoint(self.action_ckpt)

    def _load_reference_latents(self) -> None:
        data = np.load(self.reference_latents_path)
        self._ref_obs_latents = data["obs_latents"]
        self._ref_action_latents = data["action_latents"]

    def _encode_with_jax(
        self, alg, state, batch: dict, rng_key
    ) -> np.ndarray:
        import jax
        import jax.numpy as jnp

        z = alg.predict(state, batch, rng_key)
        return np.array(z)

    def encode_episodes(
        self,
        episode_states: list[np.ndarray],
        episode_actions: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode episodes to obs and action latents.

        episode_states: list of (T, obs_dim) arrays, one per episode
        episode_actions: list of (T, action_dim) arrays, one per episode
        Returns: (obs_latents, action_latents) each (N, latent_dim)
        """
        n = len(episode_states)

        if self._obs_alg is None:
            # No checkpoints loaded — return zero latents (used in testing without openx)
            return (
                np.zeros((n, 16), dtype=np.float32),
                np.zeros((n, 16), dtype=np.float32),
            )

        import jax

        rng = jax.random.key(0)

        obs_latents = []
        for i, states in enumerate(episode_states):
            obs_rng = jax.random.fold_in(rng, i)
            batch = {"observation": states[None]}
            z = self._encode_with_jax(self._obs_alg, self._obs_state, batch, obs_rng)
            obs_latents.append(z.mean(axis=0) if z.ndim > 1 else z)

        action_latents = []
        if self._action_alg is not None:
            for i, actions in enumerate(episode_actions):
                act_rng = jax.random.fold_in(rng, n + i)
                batch = {"action": actions[None]}
                z = self._encode_with_jax(self._action_alg, self._action_state, batch, act_rng)
                action_latents.append(z.mean(axis=0) if z.ndim > 1 else z)
        else:
            # Single-encoder mode: use obs encoder for both
            action_latents = obs_latents[:]

        return (
            np.stack(obs_latents).astype(np.float32),
            np.stack(action_latents).astype(np.float32),
        )

    def score_against_reference(
        self,
        obs_latents: np.ndarray,
        action_latents: np.ndarray,
        reference_obs_latents: np.ndarray,
        reference_action_latents: np.ndarray,
        k: int = 5,
    ) -> list[float]:
        """kSG MI score per episode.

        Combines cohort latents with reference pool for kNN context,
        then scores each cohort episode.
        """
        ks = np.arange(k, k + 3)

        # Pool = cohort + reference
        all_obs = np.concatenate([obs_latents, reference_obs_latents], axis=0)
        all_act = np.concatenate([action_latents, reference_action_latents], axis=0)

        scores_all = _ksg_score_np(all_obs, all_act, ks)
        # Return only the cohort episode scores (first N)
        n = len(obs_latents)
        return [float(s) for s in scores_all[:n]]

    def score_cohort(
        self,
        episode_states: list[np.ndarray],
        episode_actions: list[np.ndarray],
        k: int = 5,
    ) -> list[float]:
        """Encode + score in one call. Uses pre-loaded reference latents."""
        if self._ref_obs_latents is None:
            raise RuntimeError("Reference latents not loaded. Call _load_reference_latents first.")

        obs_lat, act_lat = self.encode_episodes(episode_states, episode_actions)
        return self.score_against_reference(
            obs_lat, act_lat,
            self._ref_obs_latents, self._ref_action_latents,
            k=k,
        )
