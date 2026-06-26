"""Robometer persistent worker — REAL-MODEL SEAM.

Written by reading:
  /data/robometer/robometer/evals/eval_server.py
  /data/robometer/scripts/example_inference.py  (the HTTP client we reuse)

Option A (server mode): eval_server.py runs as a sidecar, loads model once.
This module is a thin HTTP client around example_inference.py::compute_rewards_per_frame().

Must be imported/run inside the robometer uv env:
  cd /data/robometer && uv run python -m rdf.models.robometer_worker

DO NOT modify /data/robometer — only import from it.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from rdf.models.base import RobometerModel, RobometerScore

# Add /data/robometer to path so we can import robometer scripts directly.
_ROBOMETER_ROOT = Path("/data/robometer")
if str(_ROBOMETER_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROBOMETER_ROOT))
if str(_ROBOMETER_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROBOMETER_ROOT / "scripts"))

_MODEL_VERSION = os.environ.get("RDF_ROBOMETER_MODEL_VERSION", "Robometer-4B")


class RobometerWorker:
    """Thin HTTP client to the robometer eval_server.

    The eval_server is started separately (loads the model once).
    This class sends video frames + instruction and returns RobometerScore.
    """

    def __init__(
        self,
        server_url: str | None = None,
        fps: float = 1.0,
        timeout_s: float = 120.0,
    ):
        self.server_url = server_url or os.environ.get(
            "RDF_ROBOMETER_SERVER_URL", "http://localhost:8001"
        )
        self.fps = fps
        self.timeout_s = timeout_s
        self.model_version = _MODEL_VERSION

        # Lazy import — only works inside robometer uv env
        from example_inference import compute_rewards_per_frame, load_frames_input

        self._compute_rewards = compute_rewards_per_frame
        self._load_frames = load_frames_input

    def score_episode(self, video_path: str, instruction: str) -> RobometerScore:
        """Score one episode. Sends frames to server, returns aggregated score."""
        frames = self._load_frames(video_path, fps=self.fps)
        T = int(frames.shape[0])

        progress_array, success_array = self._compute_rewards(
            eval_server_url=self.server_url,
            video_frames=frames,
            task=instruction,
            timeout_s=self.timeout_s,
        )

        reward = float(np.mean(progress_array)) if progress_array.size else 0.0
        success_pred = float(progress_array[-1]) if success_array.size == 0 and progress_array.size else (
            float(success_array[-1]) if success_array.size else 0.0
        )

        return RobometerScore(
            reward=round(reward, 6),
            success_pred=round(success_pred, 6),
            frames_used=T,
            model_version=self.model_version,
        )

    def score_episode_from_frames(self, frames: np.ndarray, instruction: str) -> RobometerScore:
        """Score from preprocessed numpy frames (T, H, W, C) — preferred path.

        Sends frames directly to the server via /evaluate_batch_npy.
        Avoids writing a temp file and avoids re-decoding/re-preprocessing on the server.
        """
        T = int(frames.shape[0])
        progress_array, success_array = self._compute_rewards(
            eval_server_url=self.server_url,
            video_frames=frames,
            task=instruction,
            timeout_s=self.timeout_s,
        )
        reward = float(np.mean(progress_array)) if progress_array.size else 0.0
        success_pred = float(success_array[-1]) if success_array.size else (
            float(progress_array[-1]) if progress_array.size else 0.0
        )
        return RobometerScore(
            reward=round(reward, 6),
            success_pred=round(success_pred, 6),
            frames_used=T,
            model_version=self.model_version,
        )

    def score_episode_from_bytes(self, mp4_bytes: bytes, instruction: str) -> RobometerScore:
        """Fallback: write bytes to a temp file then score."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(mp4_bytes)
            tmp_path = f.name
        try:
            return self.score_episode(tmp_path, instruction)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
