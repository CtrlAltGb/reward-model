# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from `/data/reward_model`. The harness (`src/rdf/`) runs in the **robometer uv env**; DemInf scoring runs in the **openx conda env**. These are intentionally separate — never merge them.

```bash
# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Tests (no GPU needed)
/data/.conda/envs/openx/bin/python3 -m pytest tests/ -v --tb=short

# Single test
/data/.conda/envs/openx/bin/python3 -m pytest tests/test_decision.py -v --tb=short

# Full end-to-end pipeline (real models, GPU required)
/data/robometer/.venv/bin/python3 -u scripts/run_local_pipeline.py

# Smoke test (first 30 episodes only)
RDF_MAX_EPISODES=30 /data/robometer/.venv/bin/python3 -u scripts/run_local_pipeline.py
```

## Two-environment architecture

**Critical**: the pipeline spans two Python envs that cannot be merged.

| Env | Location | Used for |
|-----|----------|----------|
| robometer uv | `/data/robometer/.venv/bin/python3` | Harness, Stage A (decord, torch) |
| openx conda | `/data/.conda/envs/openx/bin/python3` | DemInf scoring (JAX, flax, TF, mcap); also runs tests |

The two envs communicate via a JSON file (`/data/reward_model_files/rdf_deminf_scores.json`). DemInf scoring writes scores once as a subprocess; the harness reads them as a lookup table. **No subprocess per episode; both models load once.**

Both upstream repos are imported by path — never install them inside this repo:
- `/data/robometer` → `sys.path` for Robometer
- `/data/demonstration-information` → `sys.path` for DemInf quality estimators

## Pipeline data flow

```
/data/clean_data/                     raw episodes (MCAP + MP4 + metadata.yaml)
        ↓
scripts/run_local_pipeline.py         ingestion → Stage A (streaming) → DemInf scoring subprocess
        ↓                                       → Stage B → decision → SQLite
/data/reward_model_files/
  rdf_integration/catalog/catalog.db  SQLite: full CatalogRow per episode
  rdf_deminf_scores.json              per-episode DemInf scores (written each run)
  rdf_pipeline_deminf/deminf_data/    episode symlinks + _cached.npz (speeds up re-runs)
  rdf_deminf_ckpts/                   pre-trained VAE checkpoint
```

`run_local_pipeline.py` loads `RobometerLocalWorker` in-process — no FastAPI server. Stage B (`infer_worker.py`) reads the pre-computed JSON scores via `DeminfWorker` — no JAX in the harness process.

To reset and re-run from scratch: `rm -rf /data/reward_model_files/rdf_integration`

## Stage A streaming execution model

Stage A uses a producer-consumer pattern:

1. **Producer thread** — `stream_episodes()` drains the episode queue, preprocesses videos in a `ThreadPoolExecutor` (8 workers), emits `(msg, manifest, frames, error)` tuples into a `threading.Queue(maxsize=8)` via `as_completed`. Puts `None` sentinel when done.
2. **Main thread** — loads `RobometerLocalWorker` concurrently while the producer runs (~70s model load).
3. **Consumer** — `run_worker(decoded_stream=...)` scores immediately when model is ready and first episode arrives. No barrier between decode and score.

Frame subsampling: `RobometerLocalWorker` reads `max_frames=8` from `exp_config.data` and subsamples with `np.linspace` before scoring. Passing more frames to the VLM slows it significantly (trained on 8 frames).

**Typical performance on 177 episodes**: Stage A ~95s (530ms/ep), DemInf ~23s (cached npz) or ~528s (first run, JAX JIT + MCAP parse), total ~190s cached / ~695s fresh.

## DemInf VAE

- State-only VAE (`obs keys: ['state']`, no images) — scores based on proprioceptive state+action KSG mutual information.
- Active checkpoint: `/data/reward_model_files/rdf_deminf_ckpts/clean_data_vae_20260626_181842/1000`
- Training code lives in `src/rdf/stage_b_deminf/train_job.py` (not called in pipeline — VAE is pre-trained). Re-training would use `scripts/train.py` in the openx env.
- MCAP topics: `/observation/state`, `/action`, `/episode/status`

## Key source locations

- `src/rdf/schemas/models.py` — all frozen Pydantic schemas (`CatalogRow`, `CohortMessage`, `EpisodeManifest`, etc.). Change here first when adding fields.
- `src/rdf/harness/catalog.py` — `LocalCatalog` (SQLite) and `AwsCatalog` (DynamoDB).
- `src/rdf/harness/queue.py` — `LocalQueue` (SQLite) and `SqsQueue`. Dedup via `dedup_id` on enqueue.
- `src/rdf/harness/video.py` — `preprocess_video()`: decodes MP4, subsamples at `i % step == 0`, center-crops to square, resizes to 256×256 at 2 fps.
- `src/rdf/models/robometer_worker.py` — `RobometerLocalWorker` (in-process checkpoint load via `load_model_from_hf`).
- `src/rdf/models/deminf_worker.py` — reads `rdf_deminf_scores.json`; `score_episodes_by_id()` bypasses MCAP entirely.
- `src/rdf/stage_b_deminf/infer_worker.py` — polls cohort queue → `score_episodes_by_id` → catalog. Checks `hasattr(model, "score_episodes_by_id")` to skip MCAP loading.
- `src/rdf/models/registry.py` — `VaeRegistry` for publishing/loading VAE checkpoints to object store.
- `configs/quality/clean_data_vae.py` — DemInf VAE training config used by `deminf_score_episodes.py`.

## Important constraints

- **Do not modify `/data/robometer` or `/data/demonstration-information`** — import only.
- **Do not add JAX/torch to the robometer venv** or vice versa — the env split is intentional.
- `accumulate_sequential` only emits a cohort for episodes where `robometer_pass is True`. If Stage A produces no passes, Stage B is skipped entirely.
- `EmbodimentConfig` defaults to synthetic topics (`/obs/rgb`, `/action`) when no yaml exists — real MCAP parsing returns empty arrays, not an error. Stage B still writes scores because `DeminfWorker.score_episodes_by_id` bypasses MCAP entirely.
