"""End-to-end pipeline runner against /data/clean_data/.

Runs all 177 episodes through Stage A (Robometer, mock) → cohort accumulation
→ Stage B (DemInf, mock) → decision, using a store adapter that reads directly
from /data/clean_data/ without copying any files.

Usage:
    cd /data/reward_model
    RDF_MODELS=mock python scripts/run_local_pipeline.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- make rdf importable without installing ---
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ.setdefault("RDF_MODELS", "mock")
os.environ.setdefault("RDF_STORAGE", "local")

import time
import uuid
import yaml
from datetime import datetime, timezone

from rdf.harness.catalog import LocalCatalog
from rdf.harness.queue import LocalQueue
from rdf.harness.storage import ObjectStore
from rdf.models.mock import MockDeminfModel, MockRobometerModel
from rdf.schemas.models import CatalogRow, CohortMessage, EpisodeManifest, PipelineConfig
from rdf.stage_a_robometer.worker import prefetch_episodes, run_worker as run_stage_a
from rdf.stage_b_deminf.infer_worker import run_infer_worker as run_stage_b
from rdf.decision.decide import decide_all

CLEAN_DATA = Path("/data/clean_data")
SCRATCH = Path("/tmp/rdf_integration")
SCRATCH.mkdir(parents=True, exist_ok=True)


class CleanDataStore(ObjectStore):
    """Read-only adapter: maps pipeline key conventions to /data/clean_data/ layout.

    Key conventions expected by the pipeline:
      raw/{episode_id}/head.mp4   ← cam_head.mp4
      raw/{episode_id}/data.mcap  ← {episode_id}.mcap
      raw/{episode_id}/metadata.yaml
    """

    def _resolve(self, key: str) -> Path | None:
        parts = key.split("/")
        if len(parts) < 3 or parts[0] != "raw":
            return None
        episode_id = parts[1]
        filename = parts[2]
        ep_dir = CLEAN_DATA / episode_id
        if filename == "head.mp4":
            return ep_dir / "cam_head.mp4"
        if filename == "data.mcap":
            # episode_XXXX.mcap
            matches = list(ep_dir.glob("*.mcap"))
            return matches[0] if matches else None
        if filename == "metadata.yaml":
            return ep_dir / "metadata.yaml"
        return None

    def get_bytes(self, key: str) -> bytes:
        p = self._resolve(key)
        if p is None or not p.exists():
            raise FileNotFoundError(key)
        return p.read_bytes()

    def exists(self, key: str) -> bool:
        p = self._resolve(key)
        return p is not None and p.exists()

    def put_bytes(self, key: str, data: bytes) -> None:
        # write to scratch for clean-bucket simulation
        dest = SCRATCH / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def delete(self, key: str) -> None:
        p = SCRATCH / key
        if p.exists():
            p.unlink()

    def list_keys(self, prefix: str) -> list[str]:
        # Only enumerate raw/ prefix for known episodes
        if prefix.startswith("raw/"):
            parts = prefix.split("/")
            if len(parts) >= 2:
                ep_dir = CLEAN_DATA / parts[1]
                if ep_dir.exists():
                    return [
                        f"raw/{parts[1]}/head.mp4",
                        f"raw/{parts[1]}/data.mcap",
                        f"raw/{parts[1]}/metadata.yaml",
                    ]
        return []


def _parse_metadata(episode_id: str) -> dict:
    meta_path = CLEAN_DATA / episode_id / "metadata.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    # Adapt field names to what EpisodeManifest expects
    tasks = meta.get("tasks", [])
    task = tasks[0] if tasks else meta.get("task", "unknown")
    return {
        "task": task,
        "instruction": task,  # no instruction field in these files
        "embodiment": meta.get("robot_type", "unknown"),
        "robot_id": meta.get("robot_id", "unknown"),
    }


def build_manifests(instruction_override: str | None = None) -> list[EpisodeManifest]:
    manifests = []
    for ep_dir in sorted(CLEAN_DATA.iterdir()):
        if not ep_dir.is_dir():
            continue
        episode_id = ep_dir.name
        fields = _parse_metadata(episode_id)
        instruction = instruction_override or fields["instruction"]
        m = EpisodeManifest(
            episode_id=episode_id,
            s3_prefix=f"raw/{episode_id}/",
            robot_id=fields["robot_id"],
            embodiment=fields["embodiment"],
            task=fields["task"],
            instruction=instruction,
            head_video_key=f"raw/{episode_id}/head.mp4",
            mcap_key=f"raw/{episode_id}/data.mcap",
            metadata_key=f"raw/{episode_id}/metadata.yaml",
            created_at=datetime.now(timezone.utc),
        )
        manifests.append(m)
    return manifests


def main():
    t_start = time.monotonic()
    print(f"\n{'='*60}")
    print("RDF Pipeline — Local Integration Test")
    print(f"Data: {CLEAN_DATA}  Scratch: {SCRATCH}")
    print(f"{'='*60}\n")

    store = CleanDataStore()
    catalog = LocalCatalog(root=str(SCRATCH / "catalog"))
    episode_queue = LocalQueue("rdf-episodes", root=str(SCRATCH / "queues"))
    cohort_queue = LocalQueue("rdf-cohorts", root=str(SCRATCH / "queues"))

    # --- Ingestion: build manifests, seed catalog, enqueue ---
    print("[ Ingestion ]")
    instruction_override = os.environ.get("RDF_INSTRUCTION")
    if instruction_override:
        print(f"  Instruction override: {instruction_override!r}")
    manifests = build_manifests(instruction_override=instruction_override)
    print(f"  Found {len(manifests)} episodes")
    now = datetime.now(timezone.utc)
    for m in manifests:
        catalog.upsert_row(CatalogRow(
            episode_id=m.episode_id,
            task=m.task,
            embodiment=m.embodiment,
            robot_id=m.robot_id,
            pipeline_mode="sequential",
            created_at=now,
            updated_at=now,
        ))
        episode_queue.enqueue(m.model_dump(mode="json"), dedup_id=m.episode_id)
    print(f"  Seeded catalog and enqueued {len(manifests)} episodes\n")

    # --- Stage A: Robometer scoring ---
    rdf_models = os.environ.get("RDF_MODELS", "mock")
    print(f"[ Stage A — Robometer ({rdf_models}) ]")
    robometer_threshold = float(os.environ.get("RDF_ROBOMETER_THRESHOLD", "0.5"))
    max_a = int(os.environ.get("RDF_MAX_EPISODES", len(manifests)))
    if max_a < len(manifests):
        print(f"  (smoke-test mode: processing {max_a}/{len(manifests)} episodes)")

    _robometer_server_proc = None
    prefetched = None

    if rdf_models == "mock":
        robometer_model = MockRobometerModel()
    else:
        import subprocess as _sp
        import threading as _threading

        _robometer_server_proc = _sp.Popen(
            [
                "/data/robometer/.venv/bin/python3",
                "robometer/evals/eval_server.py",
                "model_path=robometer/Robometer-4B",
                "batch_size=16",
                "num_gpus=1",
                "server_port=8001",
            ],
            cwd="/data/robometer",
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
        )

        # Phase 1: decode all episodes in parallel while the server loads.
        # prefetch_episodes needs no GPU — safe to run before server is ready.
        _prefetch_result: list = []
        _prefetch_exc: list = []

        def _run_prefetch():
            try:
                result = prefetch_episodes(
                    queue=episode_queue,
                    store=store,
                    catalog=catalog,
                    model_version="Robometer-4B",
                    max_episodes=max_a,
                )
                _prefetch_result.append(result)
            except Exception as exc:
                _prefetch_exc.append(exc)

        prefetch_thread = _threading.Thread(target=_run_prefetch, daemon=True)
        prefetch_thread.start()

        import requests as _req
        print("  Waiting for Robometer server …", flush=True)
        for _ in range(120):
            try:
                if _req.get("http://localhost:8001/health", timeout=5).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(5)
        else:
            _robometer_server_proc.kill()
            raise RuntimeError("Robometer server did not start in 600s")
        print("  Server ready.", flush=True)

        prefetch_thread.join()
        if _prefetch_exc:
            raise _prefetch_exc[0]
        prefetched = _prefetch_result[0]

        from rdf.models.robometer_worker import RobometerWorker
        robometer_model = RobometerWorker()

    t_a = time.monotonic()
    n_a = run_stage_a(
        model=robometer_model,
        queue=episode_queue,
        store=store,
        catalog=catalog,
        robometer_threshold=robometer_threshold,
        poll_wait=0,
        max_episodes=max_a,
        prefetched=prefetched,
    )
    elapsed_a = time.monotonic() - t_a

    # Report Stage A results — only count episodes that were actually scored
    all_tasks = list({m.task for m in manifests})
    stage_a_pass = 0
    stage_a_drop = 0
    for task in all_tasks:
        rows = catalog.rows_for_task(task)
        for r in rows:
            if r.robometer_pass is True:
                stage_a_pass += 1
            else:
                stage_a_drop += 1
    print(f"  Processed: {n_a}  Pass: {stage_a_pass}  Drop (gate): {stage_a_drop}")
    print(f"  Time: {elapsed_a:.1f}s  ({elapsed_a/n_a*1000:.0f}ms/episode)\n")

    if _robometer_server_proc is not None:
        print("  Stopping Robometer server …", flush=True)
        _robometer_server_proc.terminate()
        try:
            _robometer_server_proc.wait(timeout=30)
        except Exception:
            _robometer_server_proc.kill()
        print("  Server stopped.\n", flush=True)

    # --- Cohort accumulation (sequential mode) ---
    print("[ Cohort Accumulation ]")
    from rdf.stage_b_deminf.accumulator import accumulate_sequential
    pipeline_cfg = PipelineConfig()
    cohort_ids = accumulate_sequential(
        tasks=all_tasks,
        catalog=catalog,
        cohort_queue=cohort_queue,
        pipeline_cfg=pipeline_cfg,
        vae_version="mock-vae-v1",
        reference_set_version="mock-ref-v1",
    )
    print(f"  Tasks: {all_tasks}")
    print(f"  Cohorts emitted: {len(cohort_ids)}\n")

    # --- Stage B: DemInf scoring ---
    deminf_scores_file = os.environ.get("RDF_DEMINF_SCORES", "/tmp/rdf_deminf_scores.json")
    _use_real_deminf = Path(deminf_scores_file).exists()
    if _use_real_deminf:
        print(f"[ Stage B — DemInf (real, scores from {deminf_scores_file}) ]")
        from rdf.models.deminf_worker import DeminfWorker
        deminf_model = DeminfWorker(scores_file=deminf_scores_file)
        deminf_threshold = float(os.environ.get("RDF_DEMINF_THRESHOLD", "-10.0"))
    else:
        print("[ Stage B — DemInf (mock — no scores file found) ]")
        deminf_model = MockDeminfModel()
        deminf_threshold = 0.0

    t_b = time.monotonic()
    n_b = run_stage_b(
        model=deminf_model,
        queue=cohort_queue,
        store=store,
        catalog=catalog,
        deminf_threshold=deminf_threshold,
        poll_wait=0,
        max_cohorts=len(cohort_ids),
    )
    elapsed_b = time.monotonic() - t_b
    print(f"  Processed cohorts: {n_b}  Time: {elapsed_b:.1f}s\n")

    # --- Decision ---
    print("[ Decision ]")
    counts: dict[str, int] = {}
    for task in all_tasks:
        c = decide_all(task, catalog, robometer_threshold, deminf_threshold)
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v

    # --- Final report ---
    total = sum(counts.values())
    elapsed = time.monotonic() - t_start
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Total episodes  : {total}")
    print(f"  Keep            : {counts.get('keep', 0)}  ({counts.get('keep',0)/total*100:.1f}%)")
    print(f"  Drop            : {counts.get('drop', 0)}  ({counts.get('drop',0)/total*100:.1f}%)")
    print(f"  Pending         : {counts.get('pending', 0)}")
    print(f"  Stage A pass    : {stage_a_pass}/{total}  ({stage_a_pass/total*100:.1f}%)")
    print(f"  Stage A drop    : {stage_a_drop}/{total}  ({stage_a_drop/total*100:.1f}%)")
    print(f"  Total wall time : {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # Per-episode detail (first 20)
    print("Sample decisions (first 20 episodes):")
    print(f"  {'episode_id':<20} {'robometer':>10} {'pass':>6} {'deminf':>10} {'decision':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*6} {'-'*10} {'-'*10}")
    for task in all_tasks:
        rows = sorted(catalog.rows_for_task(task), key=lambda r: r.episode_id)
        for r in rows[:20]:
            rob = f"{r.robometer_success_pred:.4f}" if r.robometer_success_pred is not None else "N/A"
            deminf_s = f"{r.deminf_score:.4f}" if r.deminf_score is not None else "N/A"
            passed = "YES" if r.robometer_pass else "NO"
            print(f"  {r.episode_id:<20} {rob:>10} {passed:>6} {deminf_s:>10} {r.final_decision:>10}")


if __name__ == "__main__":
    main()
