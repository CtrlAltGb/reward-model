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
    RDF_DEMINF_CKPT      path to checkpoint step dir (e.g. .../1000)
    RDF_DEMINF_DATA      data root (default: /tmp/rdf_pipeline_deminf/deminf_data)
    RDF_DEMINF_SCORES    output JSON path (default: /tmp/rdf_deminf_scores.json)
    RDF_DEMINF_SPLIT     dataset split to score (default: train)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "/data/demonstration-information")
sys.path.insert(0, "/data/demonstration-information/scripts/quality")

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax.experimental import multihost_utils

tf.config.set_visible_devices([], "GPU")

import quality_estimators

CKPT = os.environ.get(
    "RDF_DEMINF_CKPT",
    sorted(Path("/tmp/rdf_deminf_ckpts").glob("*/1000"))[-1]
    if list(Path("/tmp/rdf_deminf_ckpts").glob("*/1000"))
    else "",
)
DATA_ROOT = Path(os.environ.get("RDF_DEMINF_DATA", "/tmp/rdf_pipeline_deminf/deminf_data"))
SCORES_OUT = Path(os.environ.get("RDF_DEMINF_SCORES", "/tmp/rdf_deminf_scores.json"))
SPLIT = os.environ.get("RDF_DEMINF_SPLIT", "train")


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
        estimator="ksg",
        batch_size=32,
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
