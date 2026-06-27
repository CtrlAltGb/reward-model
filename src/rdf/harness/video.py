"""Video preprocessing and frame extraction from MP4 bytes.

Preprocessing pipeline (applied before any model scoring):
  1. Decode frames at native fps, subsampling on-the-fly (never buffers all frames)
  2. Center-crop each selected frame to H×H (H = min(height, width))
  3. Resize to target_size × target_size (default 256)

Falls back to dummy arrays when av/PIL are not importable (test env).
"""

from __future__ import annotations

import io

import numpy as np

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

    Subsamples on-the-fly during decode — never materializes all frames in memory.
    For a 30 fps video at 2 fps output this processes ~1/15 of the frames through
    to_ndarray + crop + resize instead of all of them.

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
        native_fps: float = float(stream.average_rate or stream.guessed_rate or 30)

        # Decode every step-th frame — avoids buffering the whole video.
        step = max(1, int(round(native_fps / target_fps)))

        output_frames: list[np.ndarray] = []
        for i, frame in enumerate(container.decode(stream)):
            if i % step == 0:
                rgb = frame.to_ndarray(format="rgb24")
                output_frames.append(_resize(_center_crop_square(rgb), target_size))

        container.close()
    except Exception:
        return np.zeros((1, target_size, target_size, 3), dtype=np.uint8)

    if not output_frames:
        return np.zeros((1, target_size, target_size, 3), dtype=np.uint8)

    return np.stack(output_frames)


def extract_frames(mp4_bytes: bytes, n_frames: int = 8) -> np.ndarray:
    """Extract n_frames evenly-spaced frames, with preprocessing applied."""
    frames = preprocess_video(mp4_bytes)
    if len(frames) == 0:
        return np.zeros((n_frames, _TARGET_SIZE, _TARGET_SIZE, 3), dtype=np.uint8)
    indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    return np.stack([frames[i] for i in indices])
