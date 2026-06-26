"""Stage B training job — run once per task to train VAEs and save reference latents.

Calls scripts/train.py via subprocess (training is a one-off, not a hot loop).
Then encodes the reference set with DeminfWorker and saves reference_latents.npz.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from rdf.harness.logging import get_logger
from rdf.harness.storage import ObjectStore, get_object_store
from rdf.models.registry import VaeArtifact, VaeRegistry

logger = get_logger(__name__)

_DEMINF_ROOT = Path("/data/demonstration-information")
_TRAIN_SCRIPT = _DEMINF_ROOT / "scripts" / "train.py"
_DEFAULT_CONFIG = _DEMINF_ROOT / "configs" / "bc" / "manav.py:default"


def train_vae(
    task: str,
    vae_version: str,
    registry_dir: str,
    config_path: str | None = None,
    reference_mcap_keys: list[str] | None = None,
    store: ObjectStore | None = None,
) -> VaeArtifact:
    """Train VAE for a task and publish reference latents to registry.

    Args:
        task: Task name (e.g. 'pick_cup')
        vae_version: Version string (e.g. 'v1')
        registry_dir: Local path for the training output
        config_path: Path to DemInf training config
        reference_mcap_keys: Storage keys for reference MCAP files to encode
        store: ObjectStore instance
    """
    config_path = config_path or str(_DEFAULT_CONFIG)
    output_dir = Path(registry_dir) / f"task={task}" / f"vae_version={vae_version}"
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_ckpt = str(output_dir / "obs_vae")
    action_ckpt = str(output_dir / "action_vae")

    logger.info("Starting VAE training", task=task, vae_version=vae_version)
    cmd = [
        "conda", "run", "-n", "openx", "--no-capture-output",
        "python", str(_TRAIN_SCRIPT),
        f"--config={config_path}",
        f"--path={output_dir}",
        f"--name={task}_vae",
    ]
    result = subprocess.run(cmd, check=True, capture_output=False)
    logger.info("VAE training complete", returncode=result.returncode)

    # Encode reference set and save latents
    logger.info("Encoding reference set")
    ref_obs_latents, ref_action_latents = _encode_reference_set(
        obs_ckpt=obs_ckpt,
        action_ckpt=action_ckpt,
        reference_mcap_keys=reference_mcap_keys or [],
        store=store or get_object_store(),
    )

    artifact = VaeArtifact(
        task=task,
        vae_version=vae_version,
        obs_ckpt=obs_ckpt,
        action_ckpt=action_ckpt,
        reference_latents_obs=ref_obs_latents,
        reference_latents_action=ref_action_latents,
    )

    registry = VaeRegistry(store=store or get_object_store())
    registry.publish(artifact)
    logger.info("Published VAE artifact to registry", task=task, vae_version=vae_version)

    return artifact


def _encode_reference_set(
    obs_ckpt: str,
    action_ckpt: str,
    reference_mcap_keys: list[str],
    store: ObjectStore,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode reference MCAP files and return stacked latents."""
    from rdf.models.deminf_worker import DeminfWorker

    worker = DeminfWorker(obs_ckpt=obs_ckpt, action_ckpt=action_ckpt)

    if not reference_mcap_keys:
        logger.warning("No reference MCAP keys provided — using empty reference latents")
        return np.zeros((1, 16), dtype=np.float32), np.zeros((1, 16), dtype=np.float32)

    all_states = []
    all_actions = []
    for key in reference_mcap_keys:
        mcap_bytes = store.get_bytes(key)
        states = np.frombuffer(mcap_bytes, dtype=np.float32).reshape(-1, 64)
        actions = np.frombuffer(mcap_bytes[:280], dtype=np.float32).reshape(-1, 7)
        all_states.append(states)
        all_actions.append(actions)

    obs_latents, act_latents = worker.encode_episodes(all_states, all_actions)
    return obs_latents, act_latents
