"""DemInf persistent worker — REAL-MODEL SEAM.

Scoring is done by scripts/deminf_score_episodes.py (openx env), which
loads the VAE checkpoint ONCE and writes per-episode scores to a JSON file.
This class reads that JSON and serves scores to Stage B — no JAX required here.

Written by reading:
  /data/demonstration-information/scripts/quality/estimate_quality.py
  /data/demonstration-information/scripts/quality/quality_estimators.py
  /data/demonstration-information/openx/utils/evaluate.py

DO NOT modify /data/demonstration-information — only import from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rdf.harness.config import get_paths_config
from rdf.models.base import DeminfModel


class DeminfWorker:
    """Reads pre-computed per-episode scores produced by deminf_score_episodes.py.

    The scorer runs in the openx env (loads VAE checkpoint once) and writes
    scores to RDF_DEMINF_SCORES. This class is a thin lookup table over that file.
    """

    def __init__(self, scores_file: str | None = None):
        self.scores_file = Path(scores_file or get_paths_config().deminf_scores_file)
        self._scores: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not self.scores_file.exists():
            raise FileNotFoundError(
                f"DemInf scores file not found: {self.scores_file}. "
                "Run scripts/deminf_score_episodes.py first."
            )
        data = json.loads(self.scores_file.read_text())
        self._scores = data["scores"]

    def score_cohort(
        self,
        episode_states: list[np.ndarray],
        episode_actions: list[np.ndarray],
        episode_ids: list[str] | None = None,
    ) -> list[float]:
        """Return DemInf scores for a cohort.

        If episode_ids are provided, looks them up in the pre-computed scores.
        Falls back to a neutral score (0.0) for episodes not in the file.
        """
        if episode_ids is None:
            return [0.0] * len(episode_states)
        return [float(self._scores.get(eid, 0.0)) for eid in episode_ids]

    def score_episodes_by_id(self, episode_ids: list[str]) -> list[float]:
        """Direct lookup by episode_id — used by infer_worker in real mode."""
        return [float(self._scores.get(eid, 0.0)) for eid in episode_ids]

    def encode_episodes(
        self,
        episode_states: list[np.ndarray],
        episode_actions: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(episode_states)
        return (
            np.zeros((n, 18), dtype=np.float32),
            np.zeros((n, 18), dtype=np.float32),
        )

    def score_against_reference(
        self,
        obs_latents: np.ndarray,
        action_latents: np.ndarray,
        reference_obs_latents: np.ndarray,
        reference_action_latents: np.ndarray,
        k: int = 5,
    ) -> list[float]:
        n = len(obs_latents)
        return [0.0] * n
