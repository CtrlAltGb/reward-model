"""Video preprocessing and frame extraction from MP4 bytes.

Preprocessing pipeline (applied before any model scoring):
  1. Decode at native fps
  2. Center-crop each frame to H×H (H = min(height, width))
  3. Resize to target_size × target_size (default 256)
  4. Resample to target_fps (default 2 fps)

Falls back to dummy arrays when av/PIL are not importable (test env).
"""

from __future__ import annotations

import io

import numpy as np

_FALLBACK_SIZE = 256
_TARGET_FPS = 2.0
_TARGET_SIZE = 256


def _center_crop_square(frame: np.ndarray) -> np.ndarray:
    """Crop the largest centered square from (H, W, C) frame."""
    h, w = frame.shape[:2]
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    return frame[top : top + s, left : left + s]


def _resize(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize (H, W, C) uint8 frame to (size, size, C) using PIL BILINEAR."""
    from PIL import Image

    return np.array(Image.fromarray(frame).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def preprocess_video(
    mp4_bytes: bytes,
    target_fps: float = _TARGET_FPS,
    target_size: int = _TARGET_SIZE,
) -> np.ndarray:
    """Decode, center-crop, resize, and resample an MP4 to a clean frame array.

    Steps:
      1. Decode all frames from mp4_bytes at native fps.
      2. Center-crop each frame to a square (min(H,W) × min(H,W)).
      3. Resize to target_size × target_size.
      4. Subsample to target_fps by picking evenly-spaced frame indices.

    Returns:
        uint8 array of shape (T, target_size, target_size, 3).
        Falls back to a single zero frame on decode failure.
    """
    try:
        import av
    except ImportError:
        return np.zeros((1, target_size, target_size, 3), dtype=np.uint8)

    try:
        container = av.open(io.BytesIO(mp4_bytes))
        stream = container.streams.video[0]

        # Native fps — fall back to 30 if unavailable
        native_fps: float = float(stream.average_rate or stream.guessed_rate or 30)

        raw_frames: list[np.ndarray] = []
        for frame in container.decode(stream):
            raw_frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
    except Exception:
        return np.zeros((1, target_size, target_size, 3), dtype=np.uint8)

    if not raw_frames:
        return np.zeros((1, target_size, target_size, 3), dtype=np.uint8)

    # Preprocess each frame: crop → resize
    processed: list[np.ndarray] = [
        _resize(_center_crop_square(f), target_size) for f in raw_frames
    ]

    # Subsample to target_fps
    total = len(processed)
    duration_s = total / native_fps
    n_output = max(1, int(round(duration_s * target_fps)))
    indices = np.linspace(0, total - 1, n_output, dtype=int)

    return np.stack([processed[i] for i in indices])


def extract_frames(mp4_bytes: bytes, n_frames: int = 8) -> np.ndarray:
    """Extract n_frames evenly-spaced frames, with preprocessing applied.

    This is the legacy entry point used by tests and mock paths.
    For the real pipeline, prefer preprocess_video() directly.
    """
    frames = preprocess_video(mp4_bytes)
    if len(frames) == 0:
        return np.zeros((n_frames, _TARGET_SIZE, _TARGET_SIZE, 3), dtype=np.uint8)
    indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    return np.stack([frames[i] for i in indices])
