# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from `/data/reward_model`. The harness (`src/rdf/`) runs in the **robometer uv env**; DemInf scoring/training runs in the **openx conda env**. These are intentionally separate — never merge them.

```bash
# Lint
/data/.conda/envs/openx/bin/ruff check src/ tests/
/data/.conda/envs/openx/bin/ruff format --check src/ tests/

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
| openx conda | `/data/.conda/envs/openx/bin/python3` | DemInf scoring, VAE training (JAX, flax, TF, mcap); also runs tests |

The two envs communicate via a JSON file (`/data/reward_model_files/rdf_deminf_scores.json`). DemInf scoring writes scores once as a subprocess; the harness reads them as a lookup table. **No subprocess per episode; both models load once.**

Both upstream repos are imported by path — never install them inside this repo:
- `/data/robometer` → `sys.path` for Robometer
- `/data/demonstration-information` → `sys.path` for DemInf quality estimators

## Pipeline data flow

```
/data/clean_data/                          raw episodes (MCAP + MP4 + metadata.yaml)
        ↓
scripts/run_local_pipeline.py              ingestion → Stage A (streaming)
        ↓  Stage-A-pass episodes only ↓
  preprocess MCAP → _cached.npz           demonstration-information/scripts/preprocess_episodes.py
        ↓
  [auto-train VAE if no ckpt for task_id] scripts/train.py in openx env (5000 steps, 50 episodes)
        ↓
  DemInf scoring subprocess               scripts/deminf_score_episodes.py (openx env)
        ↓                                 → Stage B → decision → SQLite
/data/reward_model_files/
  rdf_integration/catalog/catalog.db      SQLite: full CatalogRow per episode
  rdf_deminf_scores.json                  per-episode DemInf scores (written each run)
  rdf_pipeline_deminf/deminf_data/
    {task_id}/train/{episode_id}/         episode symlinks (pass episodes only)
    {task_id}/train_sel/                  training subset symlinks (up to 50 episodes)
    {task_id}/test_sel/                   val subset symlinks (first 5 of train_sel)
  rdf_deminf_ckpts/
    {task_id}/clean_data_vae/{step}/      VAE checkpoint per task_id
```

`run_local_pipeline.py` loads `RobometerLocalWorker` in-process — no FastAPI server. Stage B (`infer_worker.py`) reads the pre-computed JSON scores via `DeminfWorker` — no JAX in the harness process.

**Cached `.npz` files are deleted at the end of each successful run.** They are regenerated on the next run by `preprocess_episodes.py`.

To reset and re-run from scratch: `rm -rf /data/reward_model_files/rdf_integration`

## task_id

Every episode carries a `task_id` read from `metadata.yaml`. If missing, a warning is printed and `default_task_id` from `configs/pipeline.yaml` is used (currently `"001"`).

`task_id` controls:
- Which DemInf data directory episodes land in: `deminf_data/{task_id}/train/`
- Which VAE checkpoint is used for scoring: `rdf_deminf_ckpts/{task_id}/`
- Auto-training: if no checkpoint exists for a `task_id`, a new VAE is trained automatically before scoring

## DemInf VAE auto-training

When no checkpoint exists for a `task_id`, the pipeline:
1. Creates `train_sel/` (up to `deminf_train_episodes=50` Stage-A-pass episodes) and `test_sel/` (first 5)
2. Preprocesses those episodes to `_cached.npz` via `preprocess_episodes.py`
3. Trains a BetaVAE for `deminf_train_steps=5000` steps, saving every `deminf_train_save_freq=1000` steps
4. Saves checkpoint to `rdf_deminf_ckpts/{task_id}/clean_data_vae/{step}/`

All three numbers live in `configs/models.yaml`. WandB login is required for training: `wandb login` in the openx env.

Checkpoint lookup (`_get_checkpoint_for_task`): prefers exact `deminf_train_steps` step; falls back to highest available step (so existing pre-5000-step checkpoints still work).

- Active checkpoint for `001`: `/data/reward_model_files/rdf_deminf_ckpts/001/clean_data_vae/5000`
- MCAP topics: `/observation/state`, `/action`, `/episode/status`

## Stage A streaming execution model

Stage A uses a producer-consumer pattern:

1. **Producer thread** — `stream_episodes()` drains the episode queue, preprocesses videos in a `ThreadPoolExecutor` (8 workers), emits `(msg, manifest, frames, error)` tuples into a `threading.Queue(maxsize=8)` via `as_completed`. Puts `None` sentinel when done.
2. **Main thread** — loads `RobometerLocalWorker` concurrently while the producer runs (~70s model load).
3. **Consumer** — `run_worker(decoded_stream=...)` scores immediately when model is ready and first episode arrives. No barrier between decode and score.

Frame subsampling: `RobometerLocalWorker` reads `max_frames=8` from `exp_config.data` and subsamples with `np.linspace` before scoring. Passing more frames to the VLM slows it significantly (trained on 8 frames).

Robometer checkpoint: loaded from `configs/paths.yaml::robometer_model_path` (`/data/robometer/robometer/Robometer-4B`). This is a local directory — `load_model_from_hf` skips any HF download when the path starts with `/`.

**Typical performance on 177 episodes**: Stage A ~95s (530ms/ep), DemInf ~23s (cached npz) or ~15min first run (preprocessing + JAX JIT + optional training), total ~190s cached.

## Configuration files

All paths and tuning knobs are in `configs/` — edit before running, never hardcode in source:

| File | Controls |
|------|----------|
| `configs/paths.yaml` | All filesystem paths: data dirs, checkpoints, Python envs, upstream repos |
| `configs/pipeline.yaml` | Thresholds, `default_task_id`, mode, cohort sizing, versioning |
| `configs/models.yaml` | Model versions, video settings, DemInf estimator, VAE training hyperparams |

Key env var overrides (for ad-hoc runs):
- `RDF_MAX_EPISODES` — process only first N episodes (smoke test)
- `RDF_INSTRUCTION` — override task instruction for all episodes
- `RDF_ROBOMETER_THRESHOLD` / `RDF_DEMINF_THRESHOLD` — override decision thresholds
- `RDF_DEMINF_CKPT` — force a specific checkpoint path (skips auto-detect)

## Key source locations

- `src/rdf/schemas/models.py` — all frozen Pydantic schemas (`CatalogRow`, `CohortMessage`, `EpisodeManifest`, `PathsConfig`, `ModelsConfig`, `PipelineConfig`, etc.). Change here first when adding fields.
- `src/rdf/harness/catalog.py` — `LocalCatalog` (SQLite) and `AwsCatalog` (DynamoDB).
- `src/rdf/harness/queue.py` — `LocalQueue` (SQLite) and `SqsQueue`. Dedup via `dedup_id` on enqueue.
- `src/rdf/harness/video.py` — `preprocess_video()`: decodes MP4, subsamples at `i % step == 0`, center-crops to square, resizes to 256×256 at 2 fps.
- `src/rdf/harness/config.py` — `get_paths_config()`, `get_models_config()`, `get_pipeline_config()`: `@lru_cache` loaders with YAML + env var override.
- `src/rdf/models/robometer_worker.py` — `RobometerLocalWorker` (in-process checkpoint load via `load_model_from_hf`).
- `src/rdf/models/deminf_worker.py` — reads `rdf_deminf_scores.json`; `score_episodes_by_id()` bypasses MCAP entirely.
- `src/rdf/stage_b_deminf/infer_worker.py` — polls cohort queue → `score_episodes_by_id` → catalog.
- `scripts/run_local_pipeline.py` — full pipeline: ingestion → Stage A → DemInf preprocess/train/score → Stage B → decision → cleanup.
- `scripts/deminf_score_episodes.py` — standalone DemInf scorer (openx env); auto-detects checkpoint from `rdf_deminf_ckpts/{task_id}/`.
- `configs/quality/clean_data_vae.py` — BetaVAE training config (SA joint, z=18, Franka).

## CatalogRow columns

| Column | Description |
|--------|-------------|
| `episode_id` | e.g. `episode_0001` |
| `task` | Task name from metadata |
| `embodiment` | Robot type from metadata |
| `robot_id` | Robot ID from metadata |
| `robometer_reward` | Mean per-frame progress score |
| `robometer_success_pred` | Final-frame success probability (0–1) |
| `robometer_pass` | `True` if `success_pred ≥ robometer_threshold` |
| `deminf_score` | kSG mutual-information score |
| `deminf_pass` | `True` if `deminf_score ≥ deminf_threshold` |
| `final_decision` | `"keep"` / `"drop"` / `"pending"` |
| `reasons` | Why dropped (e.g. `["task_incomplete"]`) |
| `robometer_model_version` | Model version (idempotency key) |
| `vae_version` | VAE version used |
| `pipeline_mode` | `"sequential"` or `"parallel"` |
| `created_at` / `updated_at` | Timestamps |

Note: `task_id` is on `EpisodeManifest` but not yet in `CatalogRow`.

## Important constraints

- **Do not modify `/data/robometer` or `/data/demonstration-information`** — import only.
- **Do not add JAX/torch to the robometer venv** or vice versa — the env split is intentional.
- Only Stage-A-pass episodes are forwarded to DemInf (symlinked, preprocessed, scored, trained on). Drops never touch the DemInf stage.
- `accumulate_sequential` only emits a cohort for episodes where `robometer_pass is True`. If Stage A produces no passes, Stage B is skipped entirely.
- `EmbodimentConfig` defaults to synthetic topics (`/obs/rgb`, `/action`) when no yaml exists — real MCAP parsing returns empty arrays, not an error. Stage B still writes scores because `DeminfWorker.score_episodes_by_id` bypasses MCAP entirely.
