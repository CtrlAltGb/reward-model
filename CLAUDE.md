# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from `/data/reward_model`. The harness (`src/rdf/`) runs in the **robometer uv env**; DemInf scoring runs in the **openx conda env**. These are intentionally separate.

```bash
# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Unit + integration tests (mock mode, no GPU, no upstream repos needed)
/data/.conda/envs/openx/bin/python3 -m pytest tests/ -v --tb=short

# Single test file
/data/.conda/envs/openx/bin/python3 -m pytest tests/test_e2e.py -v --tb=short

# Single test
/data/.conda/envs/openx/bin/python3 -m pytest tests/test_stage_a.py::test_worker_idempotent -v

# Full end-to-end pipeline (real models, GPU required — see README for prerequisites)
RDF_MODELS=real RDF_MAX_EPISODES=30 RDF_DEMINF_SCORES=/data/reward_model_files/rdf_deminf_scores.json \
    /data/robometer/.venv/bin/python3 -u scripts/run_local_pipeline.py
```

## Two-environment architecture

**Critical**: the pipeline spans two Python envs that cannot be merged.

| Env | Location | Used for |
|-----|----------|----------|
| robometer uv | `/data/robometer/.venv/bin/python3` | Harness, Stage A (decord, torch, fastapi) |
| openx conda | `/data/.conda/envs/openx/bin/python3` | DemInf training + scoring (JAX, flax, TF, mcap); also runs tests |

The two envs communicate via a JSON file (`/data/reward_model_files/rdf_deminf_scores.json`). DemInf scoring writes scores once; the harness reads them as a lookup table. **No subprocess per episode; both models load once.**

Both upstream repos are imported by path — never install them inside this repo:
- `/data/robometer` → `sys.path` for Robometer
- `/data/demonstration-information` → `sys.path` for DemInf quality estimators

## Pipeline data flow

```
/data/clean_data/                     raw episodes (MCAP + MP4 + metadata.yaml)
        ↓
scripts/preprocess_and_score.py       decord decode → .npz cache + Stage A scoring
        ↓
scripts/deminf_score_episodes.py      [openx env] JAX VAE scoring → /data/reward_model_files/rdf_deminf_scores.json
        ↓
scripts/run_local_pipeline.py         ingestion → Stage A → cohort → Stage B → decision → SQLite
```

`run_local_pipeline.py` (real mode) loads `RobometerLocalWorker` in-process — no FastAPI server. Stage B (`infer_worker.py`) reads the pre-computed JSON scores via `DeminfWorker` — no JAX in the harness process.

SQLite catalog lives at `/data/reward_model_files/rdf_integration/catalog/catalog.db` (persists across runs, idempotent on re-scoring). To reset and re-run from scratch: `rm -rf /data/reward_model_files/rdf_integration`.

## Stage A streaming execution model

Stage A (Robometer scoring) uses a producer-consumer pattern in real runs:

1. **Producer thread** — `stream_episodes()` drains the episode queue, fetches MP4 bytes, and preprocesses videos in a `ThreadPoolExecutor` (8 workers). Emits `(msg, manifest, frames, error)` tuples into a `threading.Queue(maxsize=8)` as each finishes (`as_completed` order). Puts `None` sentinel when done.
2. **Main thread** — loads `RobometerLocalWorker` concurrently while the producer runs.
3. **Consumer** — `run_worker(decoded_stream=...)` starts scoring the moment the model is ready and the first decoded episode arrives. No barrier between decode and score.

`run_worker()` also accepts `prefetched=` (batch mode, used in tests) or neither (inline fallback). Tests always use the inline/mock path.

## Key source locations

- `src/rdf/schemas/models.py` — all frozen Pydantic schemas (`CatalogRow`, `CohortMessage`, `EpisodeManifest`, etc.). Change here first when adding fields.
- `src/rdf/harness/catalog.py` — `LocalCatalog` (SQLite) and `AwsCatalog` (DynamoDB). The catalog stores full `CatalogRow` JSON per episode.
- `src/rdf/harness/queue.py` — `LocalQueue` (SQLite) and `SqsQueue`. Dedup via `dedup_id` on enqueue.
- `src/rdf/harness/video.py` — `preprocess_video()`: decodes MP4, subsamples on-the-fly at `i % step == 0` (never buffers all frames), center-crops to square, resizes to 256×256 at 2 fps.
- `src/rdf/models/robometer_worker.py` — `RobometerLocalWorker` (preferred: in-process checkpoint load via `load_model_from_hf`) and `RobometerWorker` (legacy HTTP client to eval_server sidecar).
- `src/rdf/models/deminf_worker.py` — reads `/data/reward_model_files/rdf_deminf_scores.json`; `score_episodes_by_id()` fast-path bypasses MCAP loading entirely.
- `src/rdf/stage_b_deminf/infer_worker.py` — polls cohort queue → calls `score_episodes_by_id` → writes to catalog. Checks `hasattr(model, "score_episodes_by_id")` first to skip MCAP loading for `DeminfWorker`.
- `src/rdf/models/registry.py` — `VaeRegistry` stores orbax checkpoints + `reference_latents.npz` in the object store under `registry/task=<task>/vae_version=<v>/`.
- `configs/embodiments/<name>.yaml` — MCAP topic names per robot. Missing file → defaults (`/obs/rgb`, `/action`). **Real topic names are not yet filled in** (`# DECISION-NEEDED` in code).

## Testing approach

All tests run in mock mode (`RDF_MODELS=mock`). The `conftest.py` sets this as default. Mocks (`src/rdf/models/mock.py`) are deterministic via seed and track `__init__` call counts to assert models load exactly once. `SyntheticMcapReader` provides fake MCAP bytes for Stage B without real files.

`tests/test_e2e.py` runs the full pipeline (ingestion → Stage A → cohort → Stage B → decision → materialize) against in-memory SQLite/local-fs backends.

## Important constraints

- **Do not modify `/data/robometer` or `/data/demonstration-information`** — import only.
- **Do not add JAX/torch to the robometer venv** or vice versa — the env split is intentional.
- The WandB stub at `/tmp/wandb_stub/wandb/__init__.py` must exist before running DemInf training (see README).
- `EmbodimentConfig` defaults to synthetic topics (`/obs/rgb`, `/action`) when no yaml exists — real MCAP parsing returns empty arrays, not an error. Stage B still writes scores because `DeminfWorker.score_episodes_by_id` bypasses MCAP entirely.
- `accumulate_sequential` only emits a cohort for episodes where `robometer_pass is True`. If Stage A produces no passes, no cohorts are emitted and Stage B is skipped.
