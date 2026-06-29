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
/data/task_2_data_real/                    raw episodes (MCAP + MP4 + metadata.yaml)
        ↓                                  episode dirs named episode_XXXX; files inside
        ↓                                  may use session_name timestamps, e.g.
        ↓                                  2026-06-24_08-37-09_0.mcap (not episode_XXXX_0.mcap)
scripts/run_local_pipeline.py              ingestion → Stage A (streaming)
        ↓  Stage-A-pass episodes only ↓
  preprocess MCAP → _cached.npz           scripts/preprocess_manav_episodes.py (openx env)
        ↓
  [auto-train two VAEs if no ckpt]        scripts/train.py x2 in openx env (50000 steps, 150 episodes)
        ↓                                 → obs_vae (state, z=7) + action_vae (action chunks, z=14)
  DemInf scoring subprocess               scripts/deminf_score_episodes.py (openx env)
        ↓                                 → Stage B → decision → SQLite
/data/reward_model_files/
  rdf_integration/catalog/catalog.db      SQLite: full CatalogRow per episode
  rdf_deminf_scores.json                  per-episode DemInf scores (written each run)
  rdf_pipeline_deminf/deminf_data/
    {task_id}/train/{episode_id}/         episode symlinks (pass episodes only)
    {task_id}/train_sel/                  training subset symlinks (up to 150 episodes)
    {task_id}/test_sel/                   val subset symlinks (first 5 of train_sel)
  rdf_deminf_ckpts/
    {task_id}/obs_vae/{step}/             observation VAE checkpoint per task_id
    {task_id}/action_vae/{step}/          action VAE checkpoint per task_id
```

`run_local_pipeline.py` loads `RobometerLocalWorker` in-process — no FastAPI server. Stage B (`infer_worker.py`) reads the pre-computed JSON scores via `DeminfWorker` — no JAX in the harness process.

**Cached `.npz` files are deleted at the end of each successful run.** They are regenerated on the next run by `preprocess_manav_episodes.py`.

To reset and re-run from scratch: `rm -rf /data/reward_model_files/rdf_integration`

## task_id

Every episode carries a `task_id` read from `metadata.yaml`. The pipeline checks both `task_id` and `id` fields (some datasets use `id: '001'` instead of `task_id`). If neither is present, a warning is printed and `default_task_id` from `configs/pipeline.yaml` is used (currently `"001"`).

`task_id` controls:
- Which DemInf data directory episodes land in: `deminf_data/{task_id}/train/`
- Which VAE checkpoints are used for scoring: `rdf_deminf_ckpts/{task_id}/obs_vae/` and `.../action_vae/`
- Auto-training: if no checkpoint exists for a `task_id`, two new VAEs are trained automatically before scoring

## Two-VAE DemInf architecture

DemInf uses **two separate VAEs** to compute kSG mutual information between observation and action latent spaces:

- **`obs_vae`** — encodes state observations (14-dim dual-arm joint+wrist pose), latent z=7
- **`action_vae`** — encodes action chunks (140-dim = chunk_size×14), latent z=14

Using a single joint SA-VAE caused degenerate scores (all -1.0) because both `obs_alg` and `action_alg` drew from the same encoder, making the latent spaces identical and the MI constant. Separate VAEs give independent latent spaces so kSG can detect true state→action correlation.

`deminf_score_episodes.py` auto-detects both checkpoint paths from `rdf_deminf_ckpts/{task_id}/obs_vae/` and `rdf_deminf_ckpts/{task_id}/action_vae/`, then passes them to `quality_estimators.get_dataset_and_score_fn`.

## DemInf VAE auto-training

When no checkpoint exists for a `task_id`, the pipeline:
1. Creates `train_sel/` (up to `deminf_train_episodes=150` Stage-A-pass episodes) and `test_sel/` (first 5)
2. Preprocesses those episodes to `_cached.npz` via `preprocess_manav_episodes.py`
3. Trains **obs_vae** and **action_vae** separately for `deminf_train_steps=50000` steps each, saving every `deminf_train_save_freq=1000` steps
4. Saves checkpoints to `rdf_deminf_ckpts/{task_id}/obs_vae/{step}/` and `.../action_vae/{step}/`

All three numbers live in `configs/models.yaml`. WandB login is required for training: `wandb login` in the openx env.

**ml_collections flag ordering constraint**: the `--config=path:cfg_str` flag MUST precede all `--config.*` overrides in the training subprocess args. Placing `--config.steps=N` before `--config=path:obs` causes a parse error.

Checkpoint lookup (`_get_checkpoint_for_task`): prefers exact `deminf_train_steps` step; falls back to highest available step (so partially-trained checkpoints still work).

- Active checkpoints for `001`: `/data/reward_model_files/rdf_deminf_ckpts/001/obs_vae/50000` and `.../action_vae/50000`
- MCAP topics (Manav): `/manav/cameras/head_cam/frame_index`, `/manav/joint_states`, `/manav/hands/left/command`, `/manav/teleop/target`

## Manav dataset format and glob fallbacks

The Manav dual-arm dataset (`/data/task_2_data_real/`) uses a naming convention where:
- Episode directories: `episode_0001/`, `episode_0002/`, etc.
- Files inside: named by **session timestamp**, e.g. `2026-06-24_08-37-09_0.mcap`, `2026-06-24_08-37-09_head_cam.mp4`

The pipeline uses glob fallbacks throughout to handle `session_name ≠ episode_id`:

- `run_local_pipeline.py` (`_setup_deminf_data`): globs `*_0.mcap` and `*_head_cam.mp4` when episode-named files don't exist
- `preprocess_manav_episodes.py`: globs `*_0.mcap` and `*_head_cam.mp4` in `process_one()`
- `run_local_pipeline.py` (`CleanDataStore._resolve`): globs `*_head_cam.mp4` as a final fallback for head.mp4

Bad episodes (empty MCAP, no frame_index messages, too short) are caught per-episode and skipped with a warning — they don't crash the preprocessing run.

**State layout** (14 dims): `[0:7]` left arm joint positions from `/manav/joint_states`, `[7:14]` left wrist pose (xyz+qxyzw) from `/manav/teleop/target`

**Action layout** (chunk_size×14 = 140 dims): `[0:6]` left hand joint commands, `[6]` left gripper, `[7:14]` left wrist target pose — repeated for `chunk_size=10` consecutive frames

NaN handling: `/manav/teleop/target` messages at episode start contain NaN (teleop controller not yet active). Only non-NaN messages are used; frames before the first valid teleop message are skipped.

## DemInf threshold: percentile-based filter

When `deminf_filter_bottom_pct > 0` in `configs/pipeline.yaml`, the fixed `deminf_threshold` is ignored and replaced with a dynamic threshold computed after scoring:

```
threshold = Nth percentile of all scored episode DemInf scores
```

This ensures exactly N% of episodes are always dropped regardless of the absolute score range. Set to `0.0` to revert to the fixed `deminf_threshold`.

Current setting: `deminf_filter_bottom_pct: 0.20` (drop worst 20%).

## Stage A streaming execution model

Stage A uses a producer-consumer pattern:

1. **Producer thread** — `stream_episodes()` drains the episode queue, preprocesses videos in a `ThreadPoolExecutor` (8 workers), emits `(msg, manifest, frames, error)` tuples into a `threading.Queue(maxsize=8)` via `as_completed`. Puts `None` sentinel when done.
2. **Main thread** — loads `RobometerLocalWorker` concurrently while the producer runs (~70s model load).
3. **Consumer** — `run_worker(decoded_stream=...)` scores immediately when model is ready and first episode arrives. No barrier between decode and score.

Frame subsampling: `RobometerLocalWorker` reads `max_frames=8` from `exp_config.data` and subsamples with `np.linspace` before scoring. Passing more frames to the VLM slows it significantly (trained on 8 frames).

Robometer checkpoint: loaded from `configs/paths.yaml::robometer_model_path` (`/data/robometer/robometer/Robometer-4B`). This is a local directory — `load_model_from_hf` skips any HF download when the path starts with `/`.

**Typical performance on ~239 episodes** (task_2_data_real): Stage A ~99s, VAE training ~267s (first run, 50k steps), DemInf scoring ~36s, total ~504s first run / ~180s cached.

## Configuration files

All paths and tuning knobs are in `configs/` — edit before running, never hardcode in source:

| File | Controls |
|------|----------|
| `configs/paths.yaml` | All filesystem paths: data dirs, checkpoints, Python envs, upstream repos |
| `configs/pipeline.yaml` | Thresholds, `default_task_id`, `default_instruction`, mode, cohort sizing, versioning |
| `configs/models.yaml` | Model versions, video settings, DemInf estimator, VAE training hyperparams |

Key current values:
- `robometer_threshold: 0.8`
- `deminf_filter_bottom_pct: 0.20`
- `default_task_id: "001"`
- `default_instruction: "Pick up the red block and place it in the green slab"`
- `deminf_train_steps: 50000`
- `deminf_train_episodes: 150`
- `clean_data_dir: /data/task_2_data_real`

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
- `src/rdf/data/manav_chunked_transform.py` — TF dataset transform for Manav chunked data; maps state(T,14) + action(T,140) to structured observation/action dicts used by DemInf.
- `scripts/run_local_pipeline.py` — full pipeline: ingestion → Stage A → DemInf preprocess/train/score → Stage B → decision → cleanup.
- `scripts/preprocess_manav_episodes.py` — Manav-specific MCAP → `_cached.npz` preprocessor (openx env); handles CDR-encoded ROS2 topics, action chunking, NaN teleop filtering.
- `scripts/deminf_score_episodes.py` — standalone DemInf scorer (openx env); auto-detects `obs_ckpt` and `action_ckpt` from `rdf_deminf_ckpts/{task_id}/obs_vae/` and `.../action_vae/`.
- `configs/quality/clean_data_vae.py` — BetaVAE training config; supports `obs`, `action`, and `sa` config_str variants for Manav chunked layout.

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

## Output files

- `/data/episode_classification.md` — per-run summary: keep/drop counts, Robometer drops, DemInf drops, threshold used
- `/data/pipeline_timings.md` — per-stage timing breakdown from a representative run

## Important constraints

- **Do not modify `/data/robometer` or `/data/demonstration-information`** — import only.
- **Do not add JAX/torch to the robometer venv** or vice versa — the env split is intentional.
- Only Stage-A-pass episodes are forwarded to DemInf (symlinked, preprocessed, scored, trained on). Drops never touch the DemInf stage.
- `accumulate_sequential` only emits a cohort for episodes where `robometer_pass is True`. If Stage A produces no passes, Stage B is skipped entirely.
- `EmbodimentConfig` defaults to synthetic topics (`/obs/rgb`, `/action`) when no yaml exists — real MCAP parsing returns empty arrays, not an error. Stage B still writes scores because `DeminfWorker.score_episodes_by_id` bypasses MCAP entirely.
- Never use `conda run` to invoke the openx env — it hangs. Always call `/data/.conda/envs/openx/bin/python3` directly.
