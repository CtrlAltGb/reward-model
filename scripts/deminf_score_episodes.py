"""
Score all episodes with the trained DemInf VAE.

Replicates estimate_quality.py logic: loads checkpoint ONCE via
quality_estimators.get_dataset_and_score_fn, runs the scoring loop,
writes per-episode scores to SCORES_OUT.

Run with:
    cd /data/demonstration-information
    /data/.conda/envs/openx/bin/python3 \
        /data/reward_model/scripts/deminf_score_episodes.py

Env vars:
    RDF_DEMINF_CKPT    path to checkpoint step dir (e.g. .../1000); overrides auto-detect
    RDF_DEMINF_DATA    override configs/paths.yaml::deminf_data_dir
    RDF_DEMINF_SCORES  override configs/paths.yaml::deminf_scores_file
    RDF_DEMINF_SPLIT   override configs/models.yaml::deminf_split
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Make rdf importable so we can use the shared config system
_RDF_SRC = Path(__file__).parent.parent / "src"
if str(_RDF_SRC) not in sys.path:
    sys.path.insert(0, str(_RDF_SRC))

from rdf.harness.config import get_models_config, get_paths_config  # noqa: E402

_paths_cfg = get_paths_config()
_models_cfg = get_models_config()

sys.path.insert(0, _paths_cfg.deminf_root)
sys.path.insert(0, str(Path(_paths_cfg.deminf_root) / "scripts" / "quality"))

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax.experimental import multihost_utils

tf.config.set_visible_devices([], "GPU")

import quality_estimators

_ckpts_dir = Path(_paths_cfg.deminf_ckpts_dir)
DATA_ROOT = Path(os.environ.get("RDF_DEMINF_DATA", _paths_cfg.deminf_data_dir))
SCORES_OUT = Path(os.environ.get("RDF_DEMINF_SCORES", _paths_cfg.deminf_scores_file))
SPLIT = os.environ.get("RDF_DEMINF_SPLIT", _models_cfg.deminf_split)

# Checkpoint: explicit env var wins; otherwise scan ckpts_dir/{task_id}/*/step.
# task_id is inferred from DATA_ROOT relative to deminf_data_dir.
def _auto_detect_ckpt() -> str:
    explicit = os.environ.get("RDF_DEMINF_CKPT")
    if explicit:
        return explicit
    try:
        task_id = DATA_ROOT.relative_to(_ckpts_dir.parent / "rdf_pipeline_deminf" / "deminf_data").parts[0]
    except ValueError:
        task_id = DATA_ROOT.name
    task_ckpt_dir = _ckpts_dir / task_id
    if not task_ckpt_dir.exists():
        return ""
    target_step = _models_cfg.deminf_train_steps
    exact = sorted(task_ckpt_dir.glob(f"*/{target_step}"))
    if exact:
        return str(exact[-1])
    all_steps = sorted(task_ckpt_dir.glob("*/[0-9]*"), key=lambda p: int(p.name))
    return str(all_steps[-1]) if all_steps else ""

CKPT = _auto_detect_ckpt()


def main():
    print(f"DemInf episode scorer", flush=True)
    print(f"  Checkpoint : {CKPT}", flush=True)
    print(f"  Data root  : {DATA_ROOT}", flush=True)
    print(f"  Split      : {SPLIT}", flush=True)
    print(f"  Output     : {SCORES_OUT}", flush=True)
    print()

    if not CKPT or not Path(CKPT).exists():
        print(f"ERROR: checkpoint not found at {CKPT}", flush=True)
        sys.exit(1)

    # ── Load checkpoint + build dataset (same as estimate_quality.py) ──────────
    # Use same checkpoint for obs and action (SA joint VAE).
    # ksg_estimator computes density-based score in joint latent space.
    print("Loading checkpoint (once) …", flush=True)
    ds, pred_fn, dataset_ids = quality_estimators.get_dataset_and_score_fn(
        estimator=_models_cfg.deminf_estimator,
        batch_size=_models_cfg.deminf_batch_size,
        obs_ckpt=str(CKPT),
        action_ckpt=str(CKPT),
        split=SPLIT,
    )
    print("Checkpoint loaded. Running scoring loop …", flush=True)

    # ── Sharding (single GPU / CPU) ────────────────────────────────────────────
    mesh = jax.sharding.Mesh(jax.devices(), axis_names="batch")
    dp_spec = jax.sharding.PartitionSpec("batch")
    dp_sharding = jax.sharding.NamedSharding(mesh, dp_spec)
    rep_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def shard(batch):
        batch = jax.tree.map(lambda x: x._numpy(), batch)
        return multihost_utils.host_local_array_to_global_array(batch, mesh, dp_spec)

    ds_sharded = map(shard, ds)
    jitted_pred = jax.jit(
        pred_fn,
        in_shardings=(dp_sharding, None),
        out_shardings=(rep_sharding, rep_sharding),
    )

    # ── Scoring loop ───────────────────────────────────────────────────────────
    rng = jax.random.key(0)
    scores = quality_estimators.estimate_quality(ds_sharded, jitted_pred, dataset_ids, rng)

    print("\nRaw scores:", flush=True)
    for ds_name, ds_scores in scores.items():
        qbi = ds_scores.get("quality_by_ep_idx", {})
        print(f"  {ds_name}: {len(qbi)} episodes scored", flush=True)

    # ── Map ep_idx → episode_id ────────────────────────────────────────────────
    # Per-episode kSG MI scores are in ds_scores["ep_idx"]: {ep_idx: mean_ksg_score}
    # Episodes are loaded in sorted order by the dataloader.
    split_dir = DATA_ROOT / SPLIT
    sorted_episode_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    idx_to_id = {i: d.name for i, d in enumerate(sorted_episode_dirs)}

    out: dict[str, float] = {}
    for ds_name, ds_scores in scores.items():
        # "ep_idx" key → {ep_idx_value: mean_ksg_score_for_that_episode}
        per_ep = ds_scores.get("ep_idx", {})
        print(f"  {ds_name}: {len(per_ep)} unique ep_idx values", flush=True)
        if not per_ep:
            print(f"  WARNING: no ep_idx scores — check dataset has ep_idx field", flush=True)
        for ep_idx, score in per_ep.items():
            ep_id = idx_to_id.get(int(ep_idx))
            if ep_id is not None:
                out[ep_id] = float(score)
            else:
                print(f"  WARNING: ep_idx={ep_idx} not in idx_to_id map (max={len(sorted_episode_dirs)-1})", flush=True)

    # ── Write output ───────────────────────────────────────────────────────────
    SCORES_OUT.parent.mkdir(parents=True, exist_ok=True)
    SCORES_OUT.write_text(json.dumps({"scores": out, "split": SPLIT, "ckpt": str(CKPT)}, indent=2))
    print(f"\nScores for {len(out)} episodes → {SCORES_OUT}", flush=True)

    if out:
        vals = list(out.values())
        print(f"  min={min(vals):.4f}  max={max(vals):.4f}  mean={sum(vals)/len(vals):.4f}", flush=True)


if __name__ == "__main__":
    main()
