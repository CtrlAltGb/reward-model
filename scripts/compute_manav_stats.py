"""
Compute dataset_statistics_openx.json from pre-generated _cached.npz files.

This replaces the broken _compute_mcap_statistics path in lerobot.py, which
reads MCAP files directly (expecting JSON-encoded topics) and fails on the
Manav native CDR MCAP format.

The output JSON format matches exactly what _compute_mcap_statistics produces,
so load_mcap_dataset can load it via the cache path when recompute_statistics=False.

Usage:
    /data/.conda/envs/openx/bin/python3 scripts/compute_manav_stats.py \
        --task_data /data/reward_model_files/rdf_pipeline_deminf/deminf_data/002 \
        --split train_sel
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/data/demonstration-information")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_data", required=True, help="Path to deminf_data/{task_id}")
    parser.add_argument("--split", default="train_sel")
    args = parser.parse_args()

    task_data = Path(args.task_data)
    split_dir = task_data / args.split
    out_path = task_data / "dataset_statistics_openx.json"

    if not split_dir.exists():
        print(f"Split dir not found: {split_dir}")
        sys.exit(1)

    import numpy as np

    # Collect all episode dirs that have a cached npz
    ep_dirs = sorted(
        e.path for e in os.scandir(split_dir)
        if e.is_dir()
    )
    ep_dirs = [d for d in ep_dirs if os.path.exists(
        os.path.join(d, os.path.basename(d) + "_cached.npz")
    )]

    if not ep_dirs:
        print(f"No cached episodes found in {split_dir}")
        sys.exit(1)

    print(f"Computing stats from {len(ep_dirs)} episodes in {split_dir} ...")

    # Running accumulators for each leaf
    # Structure matches manav_chunked_transform output (chunk_size=10):
    #   state.JOINT_POS  [7]    left arm joints (at chunk-start frame)
    #   state.MISC       [7]    left wrist pose (at chunk-start frame)
    #   action.desired_absolute.JOINT_POS  [60]  6 hand joints × 10 steps
    #   action.desired_absolute.GRIPPER    [10]  1 gripper     × 10 steps
    #   action.desired_absolute.MISC       [70]  7 wrist dims  × 10 steps

    CHUNK_SIZE = 10
    ACTION_DIM = 14

    dim_map = {
        ("state", "JOINT_POS"):                       7,
        ("state", "MISC"):                             7,
        ("action", "desired_absolute", "JOINT_POS"):  CHUNK_SIZE * 6,
        ("action", "desired_absolute", "GRIPPER"):    CHUNK_SIZE * 1,
        ("action", "desired_absolute", "MISC"):       CHUNK_SIZE * 7,
    }

    # Welford-style running stats
    counts = {k: 0 for k in dim_map}
    means  = {k: np.zeros(d, np.float64) for k, d in dim_map.items()}
    M2s    = {k: np.zeros(d, np.float64) for k, d in dim_map.items()}
    mins   = {k:  1e10 * np.ones(d, np.float64) for k, d in dim_map.items()}
    maxs   = {k: -1e10 * np.ones(d, np.float64) for k, d in dim_map.items()}

    total_steps = 0
    total_eps = 0

    for ep_dir in ep_dirs:
        ep_name = os.path.basename(ep_dir)
        npz_path = os.path.join(ep_dir, ep_name + "_cached.npz")
        data = np.load(npz_path, allow_pickle=True)
        state_arr  = data["state"].astype(np.float64)   # (T', 14)
        action_arr = data["action"].astype(np.float64)  # (T', 140)

        # Map to transform structure (exclude last step like the original)
        T = state_arr.shape[0]
        if T < 2:
            continue
        s = state_arr[:-1]   # (T'-1, 14)
        a = action_arr[:-1]  # (T'-1, 140)

        # Reshape chunked action back to (T, chunk_size, action_dim) for component extraction
        a_r = a.reshape(-1, CHUNK_SIZE, ACTION_DIM)  # (T'-1, 10, 14)
        leaf_data = {
            ("state", "JOINT_POS"):                      s[:, :7],
            ("state", "MISC"):                           s[:, 7:14],
            ("action", "desired_absolute", "JOINT_POS"): a_r[:, :, :6].reshape(-1, CHUNK_SIZE * 6),
            ("action", "desired_absolute", "GRIPPER"):   a_r[:, :, 6:7].reshape(-1, CHUNK_SIZE * 1),
            ("action", "desired_absolute", "MISC"):      a_r[:, :, 7:].reshape(-1, CHUNK_SIZE * 7),
        }

        for k, arr in leaf_data.items():
            n = arr.shape[0]
            # Welford online algorithm
            for i in range(n):
                counts[k] += 1
                delta = arr[i] - means[k]
                means[k] += delta / counts[k]
                delta2 = arr[i] - means[k]
                M2s[k] += delta * delta2
            mins[k] = np.minimum(mins[k], arr.min(axis=0))
            maxs[k] = np.maximum(maxs[k], arr.max(axis=0))

        total_steps += T - 1
        total_eps += 1

    if total_eps == 0:
        print("No valid episodes found!")
        sys.exit(1)

    def _std_tree():
        t = {"state": {}, "action": {"desired_absolute": {}}}
        for k, dim in dim_map.items():
            n = counts[k]
            val = (np.sqrt(M2s[k] / n) if n > 1 else np.ones(dim, np.float64))
            val = np.maximum(val, 1e-6).tolist()
            if len(k) == 2:
                t[k[0]][k[1]] = val
            else:
                t[k[0]][k[1]][k[2]] = val
        return t

    def _val_tree(src):
        t = {"state": {}, "action": {"desired_absolute": {}}}
        for k in dim_map:
            val = src[k].tolist()
            if len(k) == 2:
                t[k[0]][k[1]] = val
            else:
                t[k[0]][k[1]][k[2]] = val
        return t

    out = {
        "num_ep": total_eps,
        "num_steps": total_steps,
        "mean": _val_tree(means),
        "std":  _std_tree(),
        "min":  _val_tree(mins),
        "max":  _val_tree(maxs),
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Stats saved to {out_path}  ({total_eps} eps, {total_steps} steps)")


if __name__ == "__main__":
    main()
