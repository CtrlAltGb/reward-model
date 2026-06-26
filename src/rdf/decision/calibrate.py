"""Threshold calibration — sweep thresholds against a labeled validation set.

Writes configs/thresholds/<task>.yaml.
# DECISION-NEEDED: validation set format
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import yaml


class CalibrationResult(NamedTuple):
    task: str
    robometer_threshold: float
    deminf_threshold: float
    f1: float
    precision: float
    recall: float


def calibrate(
    task: str,
    episode_scores: list[dict],
    labels: list[int],
    robometer_sweep: list[float] | None = None,
    deminf_sweep: list[float] | None = None,
    output_dir: str | None = None,
) -> CalibrationResult:
    """Sweep thresholds on (score, label) pairs; pick F1-maximizing thresholds.

    episode_scores: list of {robometer_success_pred, deminf_score}
    labels: 1=keep, 0=drop (from human annotation)
    """
    import numpy as np

    robometer_sweep = robometer_sweep or [i / 20 for i in range(21)]
    deminf_sweep = deminf_sweep or [i / 20 for i in range(21)]
    labels_arr = np.array(labels)

    best: CalibrationResult | None = None

    for r_thresh in robometer_sweep:
        for d_thresh in deminf_sweep:
            preds = np.array([
                1 if (ep.get("robometer_success_pred", 0) >= r_thresh
                      and ep.get("deminf_score", 0) >= d_thresh) else 0
                for ep in episode_scores
            ])
            tp = int(np.sum((preds == 1) & (labels_arr == 1)))
            fp = int(np.sum((preds == 1) & (labels_arr == 0)))
            fn = int(np.sum((preds == 0) & (labels_arr == 1)))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            if best is None or f1 > best.f1:
                best = CalibrationResult(
                    task=task,
                    robometer_threshold=r_thresh,
                    deminf_threshold=d_thresh,
                    f1=f1,
                    precision=precision,
                    recall=recall,
                )

    assert best is not None

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(output_dir) / f"{task}.yaml"
        with open(out_path, "w") as f:
            yaml.safe_dump({
                "task": best.task,
                "robometer_threshold": best.robometer_threshold,
                "deminf_threshold": best.deminf_threshold,
                "calibrated_against": "validation_set",
                "vae_version": "v1",
                "f1": best.f1,
                "precision": best.precision,
                "recall": best.recall,
            }, f)

    return best
