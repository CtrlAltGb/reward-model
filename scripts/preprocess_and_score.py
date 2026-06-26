"""
Preprocess episodes and score with Robometer in two distinct phases.

Phase 1 — Preprocess all episodes:
  Decode cam_head.mp4 with decord, center-crop to square, resize to 256×256,
  subsample to target_fps, save as preprocessed/{episode_id}.npz.
  No GPU needed.

Phase 2 — Robometer scoring:
  Start eval server, load saved npz files, send frames, collect scores,
  stop server, symlink passing episodes for DemInf.

Run with:
    /data/robometer/.venv/bin/python3 -u scripts/preprocess_and_score.py

Env vars:
    RDF_N_EPISODES   number of episodes to process (default: 30)
    RDF_INSTRUCTION  task instruction string
    RDF_THRESHOLD    success_pred threshold (default: 0.5)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Force line-buffered output so progress is visible in background tasks
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/data/robometer")
sys.path.insert(0, "/data/robometer/scripts")

CLEAN_DATA   = Path("/data/clean_data")
SCRATCH      = Path("/tmp/rdf_pipeline_deminf")
PREPROCESSED = SCRATCH / "preprocessed"
DEMINF_DATA  = SCRATCH / "deminf_data"

N_EPISODES  = int(os.environ.get("RDF_N_EPISODES", 30))
INSTRUCTION = os.environ.get("RDF_INSTRUCTION", "pick the red cube and place in the blue box")
SERVER_URL  = "http://localhost:8001"
THRESHOLD   = float(os.environ.get("RDF_THRESHOLD", 0.5))
TARGET_FPS  = 2.0
TARGET_SIZE = 256


# ── helpers ──────────────────────────────────────────────────────────────────

def _center_crop_resize(frame: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to square then resize to (size, size, 3)."""
    from PIL import Image
    h, w = frame.shape[:2]
    s = min(h, w)
    frame = frame[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return np.array(Image.fromarray(frame).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def preprocess_video(video_path: Path, target_fps: float, target_size: int) -> np.ndarray:
    """Decode video with decord, crop, resize, subsample. Returns (T, H, W, 3) uint8."""
    import decord
    vr = decord.VideoReader(str(video_path), num_threads=2)
    native_fps = float(vr.get_avg_fps()) or 30.0
    total = len(vr)
    n_out = max(1, int(round(total * target_fps / native_fps)))
    indices = np.linspace(0, total - 1, n_out, dtype=int).tolist()
    raw = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) uint8
    del vr
    return np.stack([_center_crop_resize(raw[i], target_size) for i in range(len(raw))])


def wait_for_server(url: str, max_wait_s: int = 600) -> bool:
    import requests
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{url}/health", timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


# ── Phase 1: Preprocess ───────────────────────────────────────────────────────

def phase1_preprocess(episodes: list[Path]) -> None:
    print(f"\n{'='*60}")
    print("Phase 1: Preprocessing episodes")
    print(f"{'='*60}")
    print(f"  Episodes   : {len(episodes)}")
    print(f"  Target fps : {TARGET_FPS}")
    print(f"  Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print()

    PREPROCESSED.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    for i, ep_dir in enumerate(episodes, 1):
        eid = ep_dir.name
        out = PREPROCESSED / f"{eid}.npz"

        if out.exists():
            frames = np.load(str(out))["frames"]
            print(f"  [{i:2d}/{len(episodes)}] {eid}  cached  shape={frames.shape}")
            continue

        t_ep = time.monotonic()
        frames = preprocess_video(ep_dir / "cam_head.mp4", TARGET_FPS, TARGET_SIZE)
        np.savez_compressed(str(out), frames=frames)
        elapsed = time.monotonic() - t_ep
        print(f"  [{i:2d}/{len(episodes)}] {eid}  shape={frames.shape}  {elapsed:.1f}s")

    print(f"\n  Total: {time.monotonic()-t0:.1f}s  ({(time.monotonic()-t0)/len(episodes)*1000:.0f}ms/ep avg)")
    print("  All preprocessed frames saved.")


# ── Phase 2: Robometer scoring ────────────────────────────────────────────────

def phase2_score(episodes: list[Path]) -> dict[str, dict]:
    print(f"\n{'='*60}")
    print("Phase 2: Starting Robometer eval server")
    print(f"{'='*60}")

    server_proc = subprocess.Popen(
        [
            "/data/robometer/.venv/bin/python3",
            "robometer/evals/eval_server.py",
            "model_path=robometer/Robometer-4B",
            "batch_size=16",
            "num_gpus=1",
            "server_port=8001",
        ],
        cwd="/data/robometer",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("  Waiting for server to be ready ...")
    if not wait_for_server(SERVER_URL):
        server_proc.kill()
        raise RuntimeError("Robometer server did not start in 600s")
    print("  Server ready.")

    print(f"\n{'='*60}")
    print("  Scoring episodes")
    print(f"{'='*60}")
    print(f"  Instruction: {INSTRUCTION!r}")
    print(f"  Threshold  : {THRESHOLD}")
    print()

    from rdf.models.robometer_worker import RobometerWorker
    worker = RobometerWorker(server_url=SERVER_URL)

    results: dict[str, dict] = {}
    t0 = time.monotonic()

    for i, ep_dir in enumerate(episodes, 1):
        eid = ep_dir.name
        frames = np.load(str(PREPROCESSED / f"{eid}.npz"))["frames"]
        t_ep = time.monotonic()
        score = worker.score_episode_from_frames(frames, INSTRUCTION)
        ms = (time.monotonic() - t_ep) * 1000
        passed = score.success_pred >= THRESHOLD
        results[eid] = {
            "reward": score.reward,
            "success_pred": score.success_pred,
            "passed": passed,
            "latency_ms": round(ms, 1),
        }
        tag = "PASS" if passed else "FAIL"
        print(
            f"  [{i:2d}/{len(episodes)}] {eid}  "
            f"reward={score.reward:.4f}  success={score.success_pred:.4f}  "
            f"[{tag}]  ({ms:.0f}ms)"
        )

    passing = [ep for ep, r in results.items() if r["passed"]]
    total_s = time.monotonic() - t0
    print(f"\n  Passed : {len(passing)}/{len(episodes)}")
    print(f"  Time   : {total_s:.1f}s  ({total_s/len(episodes)*1000:.0f}ms/ep avg)")

    print(f"\n{'='*60}")
    print("  Stopping Robometer eval server")
    print(f"{'='*60}")
    server_proc.terminate()
    try:
        server_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server_proc.kill()
    print("  Server stopped. GPU released.")

    return results


# ── Phase 3: Symlink for DemInf ───────────────────────────────────────────────

def phase3_symlink(results: dict[str, dict]) -> None:
    print(f"\n{'='*60}")
    print("Phase 3: Symlinking passing episodes for DemInf")
    print(f"{'='*60}")

    passing = [ep for ep, r in results.items() if r["passed"]]
    if not passing:
        print("  No episodes passed — nothing to symlink.")
        return

    n_train = max(1, int(len(passing) * 0.8))
    train_eps = passing[:n_train]
    test_eps  = passing[n_train:] if len(passing) > n_train else [passing[-1]]

    for split, eps in [("train", train_eps), ("test", test_eps)]:
        split_dir = DEMINF_DATA / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for eid in eps:
            ep_link_dir = split_dir / eid
            ep_link_dir.mkdir(exist_ok=True)
            src_dir = CLEAN_DATA / eid
            for mcap_src in src_dir.glob("*.mcap"):
                dst = ep_link_dir / mcap_src.name
                if not dst.exists():
                    dst.symlink_to(mcap_src.resolve())
            for vid in ["cam_head.mp4", "cam_wrist.mp4"]:
                src = src_dir / vid
                dst = ep_link_dir / vid
                if src.exists() and not dst.exists():
                    dst.symlink_to(src.resolve())

    print(f"  Train : {len(train_eps)} episodes")
    print(f"  Test  : {len(test_eps)} episodes")
    print(f"  Dir   : {DEMINF_DATA}")

    info = {
        "deminf_data_dir": str(DEMINF_DATA),
        "train_episodes": train_eps,
        "test_episodes": test_eps,
    }
    results_blob = {
        "results": results,
        "passing": passing,
        "instruction": INSTRUCTION,
        **info,
    }
    (SCRATCH / "results.json").write_text(json.dumps(results_blob, indent=2))
    print(f"  Results → {SCRATCH}/results.json")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    episodes = sorted([d for d in CLEAN_DATA.iterdir() if d.is_dir()])[:N_EPISODES]

    phase1_preprocess(episodes)
    results = phase2_score(episodes)
    phase3_symlink(results)

    print(f"\n{'='*60}")
    print("Done. Next steps for DemInf training:")
    print(f"  cd /data/demonstration-information")
    print(f"  /data/.conda/envs/openx/bin/python3 scripts/preprocess_episodes.py \\")
    print(f"      --root {DEMINF_DATA} --splits train test --workers 4")
    print(f"  /data/.conda/envs/openx/bin/python3 scripts/train.py \\")
    print(f"      --config /data/reward_model/configs/quality/clean_data_vae.py:sa \\")
    print(f"      --path /tmp/rdf_deminf_ckpts --name clean_data_vae")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
