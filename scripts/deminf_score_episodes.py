"""
Score all episodes with trained DemInf VAEs (separate obs and action checkpoints).

Replicates estimate_quality.py logic: loads checkpoints ONCE via
quality_estimators.get_dataset_and_score_fn with distinct obs_ckpt and action_ckpt,
runs the scoring loop, writes per-episode scores to SCORES_OUT.

Run with:
    cd /data/demonstration-information
    /data/.conda/envs/openx/bin/python3 \\
        /data/reward_model/scripts/deminf_score_episodes.py

Env vars:
    RDF_DEMINF_OBS_CKPT     path to obs_vae checkpoint step dir; overrides auto-detect
    RDF_DEMINF_ACTION_CKPT  path to action_vae checkpoint step dir; overrides auto-detect
    RDF_DEMINF_CKPT         legacy: use same checkpoint for both obs and action
    RDF_DEMINF_DATA         override configs/paths.yaml::deminf_data_dir
    RDF_DEMINF_SCORES       override configs/paths.yaml::deminf_scores_file
    RDF_DEMINF_SPLIT        override configs/models.yaml::deminf_split
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

_RDF_SRC = Path(__file__).parent.parent / "src"
if str(_RDF_SRC) not in sys.path:
    sys.path.insert(0, str(_RDF_SRC))

from rdf.harness.config import get_models_config, get_paths_config  # noqa: E402

_paths_cfg = get_paths_config()
_models_cfg = get_models_config()

sys.path.insert(0, _paths_cfg.deminf_root)
sys.path.insert(0, str(Path(_paths_cfg.deminf_root) / "scripts" / "quality"))

import jax
import tensorflow as tf
from jax.experimental import multihost_utils

tf.config.set_visible_devices([], "GPU")

import quality_estimators

_ckpts_dir = Path(_paths_cfg.deminf_ckpts_dir)
DATA_ROOT = Path(os.environ.get("RDF_DEMINF_DATA", _paths_cfg.deminf_data_dir))
SCORES_OUT = Path(os.environ.get("RDF_DEMINF_SCORES", _paths_cfg.deminf_scores_file))
SPLIT = os.environ.get("RDF_DEMINF_SPLIT", _models_cfg.deminf_split)


def _auto_detect_ckpts() -> tuple[str, str]:
    """Return (obs_ckpt, action_ckpt).  Legacy single-ckpt env var maps to both."""
    legacy = os.environ.get("RDF_DEMINF_CKPT", "")
    if legacy:
        return legacy, legacy

    obs_explicit = os.environ.get("RDF_DEMINF_OBS_CKPT", "")
    action_explicit = os.environ.get("RDF_DEMINF_ACTION_CKPT", "")
    if obs_explicit or action_explicit:
        return obs_explicit, action_explicit

    # Auto-detect from task_id embedded in DATA_ROOT path
    try:
        task_id = DATA_ROOT.relative_to(
            _ckpts_dir.parent / "rdf_pipeline_deminf" / "deminf_data"
        ).parts[0]
    except ValueError:
        task_id = DATA_ROOT.name

    task_ckpt_dir = _ckpts_dir / task_id

    def _best(subdir: str) -> str:
        d = task_ckpt_dir / subdir
        if not d.exists():
            return ""
        target = d / str(_models_cfg.deminf_train_steps)
        if target.exists():
            return str(target)
        steps = sorted(
            [p for p in d.iterdir() if p.name.isdigit()],
            key=lambda p: int(p.name),
        )
        return str(steps[-1]) if steps else ""

    return _best("obs_vae"), _best("action_vae")


OBS_CKPT, ACTION_CKPT = _auto_detect_ckpts()


def main():
    print("DemInf episode scorer (separate obs + action VAEs)", flush=True)
    print(f"  Obs checkpoint    : {OBS_CKPT}", flush=True)
    print(f"  Action checkpoint : {ACTION_CKPT}", flush=True)
    print(f"  Data root         : {DATA_ROOT}", flush=True)
    print(f"  Split             : {SPLIT}", flush=True)
    print(f"  Output            : {SCORES_OUT}", flush=True)
    print()

    if not OBS_CKPT or not Path(OBS_CKPT).exists():
        print(f"ERROR: obs checkpoint not found at {OBS_CKPT!r}", flush=True)
        sys.exit(1)
    if not ACTION_CKPT or not Path(ACTION_CKPT).exists():
        print(f"ERROR: action checkpoint not found at {ACTION_CKPT!r}", flush=True)
        sys.exit(1)

    print("Loading checkpoints (once) …", flush=True)
    ds, pred_fn, dataset_ids = quality_estimators.get_dataset_and_score_fn(
        estimator=_models_cfg.deminf_estimator,
        batch_size=_models_cfg.deminf_batch_size,
        obs_ckpt=OBS_CKPT,
        action_ckpt=ACTION_CKPT,
        split=SPLIT,
    )
    print("Checkpoints loaded. Running scoring loop …", flush=True)

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

    rng = jax.random.key(0)
    scores = quality_estimators.estimate_quality(ds_sharded, jitted_pred, dataset_ids, rng)

    print("\nRaw scores:", flush=True)
    for ds_name, ds_scores in scores.items():
        qbi = ds_scores.get("quality_by_ep_idx", {})
        print(f"  {ds_name}: {len(qbi)} episodes scored", flush=True)

    split_dir = DATA_ROOT / SPLIT
    sorted_episode_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    idx_to_id = {i: d.name for i, d in enumerate(sorted_episode_dirs)}

    out: dict[str, float] = {}
    for ds_name, ds_scores in scores.items():
        per_ep = ds_scores.get("ep_idx", {})
        print(f"  {ds_name}: {len(per_ep)} unique ep_idx values", flush=True)
        if not per_ep:
            print("  WARNING: no ep_idx scores — check dataset has ep_idx field", flush=True)
        for ep_idx, score in per_ep.items():
            ep_id = idx_to_id.get(int(ep_idx))
            if ep_id is not None:
                out[ep_id] = float(score)
            else:
                print(
                    f"  WARNING: ep_idx={ep_idx} not in idx_to_id map"
                    f" (max={len(sorted_episode_dirs)-1})",
                    flush=True,
                )

    SCORES_OUT.parent.mkdir(parents=True, exist_ok=True)
    SCORES_OUT.write_text(
        json.dumps(
            {"scores": out, "split": SPLIT, "obs_ckpt": OBS_CKPT, "action_ckpt": ACTION_CKPT},
            indent=2,
        )
    )
    print(f"\nScores for {len(out)} episodes → {SCORES_OUT}", flush=True)

    if out:
        vals = list(out.values())
        mean = sum(vals) / len(vals)
        print(f"  min={min(vals):.4f}  max={max(vals):.4f}  mean={mean:.4f}", flush=True)


if __name__ == "__main__":
    main()
