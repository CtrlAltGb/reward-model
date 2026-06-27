"""
BetaVAE trained on Robometer-filtered clean_data episodes (Franka MCAP format).

Data layout expected at configs/paths.yaml::deminf_data_dir:
    train/episode_XXXX/{episode_XXXX.mcap, cam_head.mp4}
    test/episode_XXXX/{episode_XXXX.mcap, cam_head.mp4}

Usage (from /data/demonstration-information):
    /data/.conda/envs/openx/bin/python3 scripts/train.py \\
        --config /data/reward_model/configs/quality/clean_data_vae.py:sa \\
        --path /tmp/rdf_deminf_ckpts \\
        --name clean_data_vae

config_str options:
    s   — obs state VAE only  (z_dim=12)
    a   — action VAE only     (z_dim=6)
    sa  — joint state+action  (z_dim=18)
"""

import os

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.beta_vae import BetaVAE
from openx.data.datasets.manav import manav_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.components.mlp import MLP
from openx.networks.core import Concatenate, MultiDecoder, MultiEncoder
from openx.utils.spec import ModuleSpec

DEMINF_DATA = os.environ.get("RDF_DEMINF_DATA", "/data/reward_model_files/rdf_pipeline_deminf/deminf_data")


def get_config(config_str: str = "sa"):
    config_type = config_str
    assert config_type in {"s", "a", "sa"}, f"Unknown config_str {config_str!r}"

    vae_keys = {
        "s":  {"observation->state": None},
        "a":  {"action": None},
        "sa": {"observation->state": None, "action": None},
    }[config_type]
    z_dim = {"s": 12, "a": 6, "sa": 18}[config_type]

    structure = {
        "observation": {
            "state": {
                StateEncoding.JOINT_POS: NormalizationType.GAUSSIAN,
                StateEncoding.MISC:      NormalizationType.GAUSSIAN,
            },
        },
        "action": {
            "desired_absolute": {
                StateEncoding.JOINT_POS: NormalizationType.GAUSSIAN,
                StateEncoding.GRIPPER:   NormalizationType.BOUNDS,
                StateEncoding.MISC:      NormalizationType.GAUSSIAN,
            },
        },
    }

    dataloader = dict(
        datasets={
            "clean_data": dict(
                path=DEMINF_DATA,
                train_split="train",
                val_split="test",
                transform=ModuleSpec.create(manav_dataset_transform),
            ),
        },
        n_obs=1,
        n_action=1,
        shuffle_size=2000,
        batch_size=32,
        recompute_statistics=True,
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
