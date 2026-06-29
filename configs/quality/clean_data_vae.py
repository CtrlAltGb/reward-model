"""
BetaVAE trained on Robometer-filtered clean_data episodes (Manav dual-arm, chunked actions).

State (14 dims): left arm joints [0:7] + left wrist pose [7:14]
Action (140 dims): 10-step action chunks, time-first flattened
    JOINT_POS: 10 × 6  = 60   left hand joint commands per step
    GRIPPER:   10 × 1  = 10   left gripper per step
    MISC:      10 × 7  = 70   left wrist target pose per step

Two separate VAEs are trained — one for obs state, one for action chunks — so that
kSG mutual information estimation gets genuinely different latent spaces (z_obs ≠ z_action).

Usage (from /data/demonstration-information):
    /data/.conda/envs/openx/bin/python3 scripts/train.py \\
        --config /data/reward_model/configs/quality/clean_data_vae.py:obs \\
        --path /tmp/rdf_deminf_ckpts --name obs_vae --include_timestamp=False

config_str options:
    obs    — observation state VAE only  (z_dim=7)
    action — action chunk VAE only       (z_dim=14)
    sa     — joint state+action (backward compat, z_dim=14)
"""

import os  # noqa: I001
import sys
from pathlib import Path

# Make rdf.data importable when this config is loaded by train.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import optax
import tensorflow as tf
from ml_collections import ConfigDict
from openx.algs.beta_vae import BetaVAE
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.components.mlp import MLP
from openx.networks.core import Concatenate, MultiDecoder, MultiEncoder
from openx.utils.spec import ModuleSpec

from rdf.data.manav_chunked_transform import manav_chunked_transform

DEMINF_DATA = os.environ.get(
    "RDF_DEMINF_DATA",
    "/data/reward_model_files/rdf_pipeline_deminf/deminf_data",
)


def get_config(config_str: str = "obs"):
    assert config_str in {"obs", "action", "sa"}, f"Unknown config_str {config_str!r}"

    vae_keys = {
        "obs":    {"observation->state": None},
        "action": {"action": None},
        "sa":     {"observation->state": None, "action": None},
    }[config_str]
    z_dim = {"obs": 7, "action": 14, "sa": 14}[config_str]

    # Full structure needed for normalization of all keys, even when only a subset is encoded.
    structure = {
        "observation": {
            "state": {
                StateEncoding.JOINT_POS: NormalizationType.GAUSSIAN,  # [7]
                StateEncoding.MISC:      NormalizationType.GAUSSIAN,  # [7]
            },
        },
        "action": {
            "desired_absolute": {
                StateEncoding.JOINT_POS: NormalizationType.GAUSSIAN,  # [60] = 10 × 6
                StateEncoding.GRIPPER:   NormalizationType.BOUNDS,    # [10] = 10 × 1
                StateEncoding.MISC:      NormalizationType.GAUSSIAN,  # [70] = 10 × 7
            },
        },
    }

    dataloader = dict(
        datasets={
            "clean_data": dict(
                path=DEMINF_DATA,
                train_split="train_sel",
                val_split="test_sel",
                transform=ModuleSpec.create(manav_chunked_transform),
            ),
        },
        n_obs=1,
        n_action=1,
        shuffle_size=2000,
        batch_size=32,
        recompute_statistics=False,
        cache=True,
        prefetch=tf.data.AUTOTUNE,
    )

    alg = ModuleSpec.create(
        BetaVAE,
        encoder=ModuleSpec.create(
            MultiEncoder,
            encoders=vae_keys,
            trunk=ModuleSpec.create(
                Concatenate,
                model=ModuleSpec.create(MLP, [512, 512], activate_final=True),
                flatten_time=True,
            ),
        ),
        decoder=ModuleSpec.create(
            MultiDecoder,
            trunk=ModuleSpec.create(MLP, [512, 512], activate_final=True),
            decoders=vae_keys,
        ),
        z_dim=z_dim,
        beta=0.05,
    )

    return ConfigDict(
        dict(
            structure=structure,
            alg=alg,
            dataloader=dataloader,
            optimizer=ModuleSpec.create(optax.adam),
            lr_schedule=ModuleSpec.create(optax.constant_schedule, 0.0001),
            steps=1000,
            log_freq=50,
            val_freq=200,
            save_freq=250,
            val_steps=5,
            seed=42,
        )
    )
