"""Robometer persistent worker — REAL-MODEL SEAM.

Two implementations:

RobometerLocalWorker (preferred)
    Loads the checkpoint in-process via load_model_from_hf, calls
    compute_batch_outputs directly.  No server, no HTTP, no startup wait.
    Mirrors scripts/example_inference_local.py.

RobometerWorker (legacy)
    Thin HTTP client to a separately-started eval_server.py sidecar.
    Kept for backwards compatibility / multi-GPU server use.

DO NOT modify /data/robometer — only import from it.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from rdf.harness.config import get_models_config, get_paths_config
from rdf.models.base import RobometerModel, RobometerScore

_paths_cfg = get_paths_config()
_models_cfg = get_models_config()

# Add robometer repo to path so we can import robometer scripts directly.
_ROBOMETER_ROOT = Path(_paths_cfg.robometer_root)
if str(_ROBOMETER_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROBOMETER_ROOT))
if str(_ROBOMETER_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROBOMETER_ROOT / "scripts"))

_MODEL_VERSION = _models_cfg.robometer_model_version


class RobometerWorker:
    """Thin HTTP client to the robometer eval_server.

    The eval_server is started separately (loads the model once).
    This class sends video frames + instruction and returns RobometerScore.
    """

    def __init__(
        self,
        server_url: str | None = None,
        fps: float | None = None,
        timeout_s: float | None = None,
    ):
        self.server_url = server_url or _models_cfg.robometer_server_url
        self.fps = fps if fps is not None else _models_cfg.robometer_server_fps
        self.timeout_s = timeout_s if timeout_s is not None else _models_cfg.robometer_server_timeout_s
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


class RobometerLocalWorker:
    """Loads Robometer checkpoint in-process — no FastAPI server required.

    Mirrors scripts/example_inference_local.py but loads once and scores many.
    Eliminates ~60s server startup vs RobometerWorker.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
    ):
        import torch
        from robometer.data.dataset_types import ProgressSample, Trajectory
        from robometer.evals.eval_server import compute_batch_outputs
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        self.model_version = _MODEL_VERSION
        self._model_path = model_path or _paths_cfg.robometer_model_path
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        exp_config, tokenizer, processor, reward_model = load_model_from_hf(
            model_path=self._model_path,
            device=self._device,
        )
        reward_model.eval()

        self._model = reward_model
        self._tokenizer = tokenizer
        self._batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
        self._compute_batch_outputs = compute_batch_outputs
        self._Trajectory = Trajectory
        self._ProgressSample = ProgressSample

        loss_config = getattr(exp_config, "loss", None)
        self._is_discrete = (
            getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
            if loss_config else False
        )
        self._num_bins = (
            getattr(loss_config, "progress_discrete_bins", None)
            or getattr(exp_config.model, "progress_discrete_bins", 10)
        )
        # Model was trained on max_frames frames — subsample to this count at inference
        # to match training distribution and avoid 3-5× token overhead from longer videos.
        self._max_frames: int = getattr(exp_config.data, "max_frames", 8) or 8

    def score_episode_from_frames(self, frames: np.ndarray, instruction: str) -> RobometerScore:
        import torch

        # Subsample to max_frames evenly spaced across the episode.
        if len(frames) > self._max_frames:
            idx = np.linspace(0, len(frames) - 1, self._max_frames, dtype=int)
            frames = frames[idx]

        T = int(frames.shape[0])
        traj = self._Trajectory(
            frames=frames,
            frames_shape=tuple(frames.shape),
            task=instruction,
            id="0",
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        sample = self._ProgressSample(trajectory=traj, sample_type="progress")
        batch = self._batch_collator([sample])

        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(self._device)

        with torch.no_grad():
            results = self._compute_batch_outputs(
                self._model,
                self._tokenizer,
                progress_inputs,
                sample_type="progress",
                is_discrete_mode=self._is_discrete,
                num_bins=self._num_bins,
            )

        progress_pred = results.get("progress_pred", [])
        progress_array = (
            np.array(progress_pred[0], dtype=np.float32)
            if progress_pred else np.array([], dtype=np.float32)
        )
        outputs_success = results.get("outputs_success", {}) or {}
        success_probs = outputs_success.get("success_probs", [])
        success_array = (
            np.array(success_probs[0], dtype=np.float32)
            if success_probs else np.array([], dtype=np.float32)
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
