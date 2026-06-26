# Robot Data-Filtering Pipeline

A two-stage pipeline for filtering robot demonstration episodes before they enter training. Stage A gates on task completion (Robometer reward model). Stage B trims low-quality demonstrations (DemInf mutual-information VAE score). Only episodes that pass both stages are written to the clean bucket.

---

## Pipeline overview

```
/data/clean_data/   (raw MCAP + MP4 episodes)
        │
        ▼
[ Preprocessing ]             scripts/preprocess_and_score.py — Phase 1
  decord decode → center-crop → 256×256 → 2fps → save .npz
  Output: /tmp/rdf_pipeline_deminf/preprocessed/{episode_id}.npz
        │
        ▼
[ Stage A — Robometer ]       scripts/preprocess_and_score.py — Phase 2
                              OR scripts/run_local_pipeline.py with RDF_MODELS=real
  Robometer-4B eval server (HTTP, port 8001) — model loaded ONCE
  score_episode_from_frames(frames, instruction) → success_pred
  gate: success_pred ≥ 0.5 → pass / drop(task_incomplete)
        │  pass ↓
        ▼
[ DemInf Training ]           scripts/run_with_deminf.sh  (run once after Stage A)
  Passing episodes symlinked → /tmp/rdf_pipeline_deminf/deminf_data/{train,test}/
  preprocess_episodes.py caches MCAP → .npz
  scripts/train.py trains BetaVAE (SA joint, z=18) for 1000 steps
  Checkpoint → /tmp/rdf_deminf_ckpts/<run_name>/1000/
        │
        ▼
[ DemInf Scoring ]            scripts/deminf_score_episodes.py
  Runs in openx conda env — checkpoint loaded ONCE
  Replicates estimate_quality.py: ksg_estimator over train+test splits
  Per-episode scores → /tmp/rdf_deminf_scores.json
        │
        ▼
[ Stage B — DemInf ]          run_local_pipeline.py — DeminfWorker (lookup table)
  Reads /tmp/rdf_deminf_scores.json
  Scores cohort of passing episodes → deminf_score per episode
  trim: deminf_score ≥ deminf_threshold → pass / drop(low_quality_jitter)
        │
        ▼
[ Decision ]                  decide_all() — pure function, no model calls
  Robometer gate + DemInf trim → keep / drop / pending
        ▼
Clean episodes (for training)
```

---

## Actual run results (2026-06-26)

| Stage | Episodes | Pass | Drop | Time |
|---|---|---|---|---|
| Preprocessing (decord) | 30 | — | — | ~2.9s/ep |
| Stage A — Robometer | 30 | 30 | 0 | ~8.7s/ep (decode + inference) |
| DemInf training | 24 train / 6 test | — | — | 1000 steps, ~3 min |
| DemInf scoring | 24 train + 6 test | — | — | ~3s total (JAX JIT) |
| Stage B | 30 (1 cohort) | 30 | 0 | <0.1s |
| **Total wall time** | **30 episodes** | **30 keep** | **0 drop** | **~6 min** |

Robometer scores ranged from `success_pred=0.86–0.97`. All 30 clean episodes passed both gates, as expected for a high-quality reference dataset.

---

## Scripts inventory

| Script | Env | Purpose |
|---|---|---|
| `scripts/preprocess_and_score.py` | robometer uv | Phase 1: decode all videos → .npz cache. Phase 2: start Robometer server → score all → Phase 3: symlink passing eps for DemInf |
| `scripts/run_with_deminf.sh` | bash wrapper | Runs preprocess_and_score.py → preprocess_episodes.py → train.py end-to-end |
| `scripts/deminf_score_episodes.py` | openx conda | Loads DemInf checkpoint ONCE, scores all episodes via kSG estimator, writes `/tmp/rdf_deminf_scores.json` |
| `scripts/run_local_pipeline.py` | robometer uv | Full harness pipeline: ingestion → Stage A (starts/stops Robometer server) → cohort accumulation → Stage B → decision |

---

## Full end-to-end run (from scratch)

### Step 1 — Preprocess + Stage A + prepare DemInf data

```bash
cd /data/reward_model
/data/robometer/.venv/bin/python3 -u scripts/preprocess_and_score.py
```

This:
- Decodes 30 episodes with decord → `preprocessed/{ep}.npz`
- Starts Robometer eval server, scores all episodes, stops server
- Symlinks passing episodes to `deminf_data/{train,test}/`

### Step 2 — Preprocess MCAP + train DemInf VAE

```bash
cd /data/demonstration-information

# Cache MCAP → .npz for fast training
/data/.conda/envs/openx/bin/python3 scripts/preprocess_episodes.py \
    --root /tmp/rdf_pipeline_deminf/deminf_data \
    --splits train test \
    --workers 4

# Train BetaVAE (joint state+action, z=18)
PYTHONPATH=/tmp/wandb_stub \
/data/.conda/envs/openx/bin/python3 scripts/train.py \
    --config /data/reward_model/configs/quality/clean_data_vae.py:sa \
    --path /tmp/rdf_deminf_ckpts \
    --name clean_data_vae
```

Checkpoint saved under `/tmp/rdf_deminf_ckpts/clean_data_vae_<timestamp>/1000/`.

### Step 3 — Score all episodes with the DemInf VAE

```bash
cd /data/demonstration-information

# Score train split
RDF_DEMINF_CKPT=/tmp/rdf_deminf_ckpts/clean_data_vae_<timestamp>/1000 \
RDF_DEMINF_DATA=/tmp/rdf_pipeline_deminf/deminf_data \
RDF_DEMINF_SCORES=/tmp/rdf_deminf_scores.json \
RDF_DEMINF_SPLIT=train \
PYTHONPATH=/tmp/wandb_stub \
/data/.conda/envs/openx/bin/python3 -u \
    /data/reward_model/scripts/deminf_score_episodes.py
```

Per-episode kSG MI scores written to `/tmp/rdf_deminf_scores.json`.

### Step 4 — Run the full harness pipeline

```bash
cd /data/reward_model

RDF_MODELS=real \
RDF_MAX_EPISODES=30 \
RDF_DEMINF_SCORES=/tmp/rdf_deminf_scores.json \
RDF_DEMINF_THRESHOLD=-10.0 \
/data/robometer/.venv/bin/python3 -u scripts/run_local_pipeline.py
```

This automatically starts and stops the Robometer eval server for Stage A, then uses the pre-computed DemInf scores for Stage B.

---

## Source layout

```
src/rdf/
  schemas/
    models.py           Frozen Pydantic v2 schemas: EpisodeManifest, RobometerResult,
                        CohortMessage, DeminfResult, CatalogRow, PipelineConfig, ...
  harness/
    catalog.py          LocalCatalog (SQLite) + AwsCatalog (DynamoDB)
    queue.py            LocalQueue (SQLite) + SqsQueue
    storage.py          ObjectStore: LocalObjectStore + S3ObjectStore
    logging.py          structlog JSON logging with episode/cohort context
    mcap_extract.py     MCAP → (states, actions); SyntheticMcapReader for tests
    video.py            MP4 frame extraction (decord / PyAV)
    config.py           Load configs/embodiments/*.yaml, configs/thresholds/*.yaml
    idempotency.py      Skip re-scoring already-scored episodes
  models/
    base.py             Protocol interfaces: RobometerModel, DeminfModel
    mock.py             Deterministic mock implementations (no GPU, no JAX)
    robometer_worker.py Thin HTTP client to the Robometer eval_server
                        score_episode_from_frames() → uses cached .npz, no redecode
    deminf_worker.py    Reads /tmp/rdf_deminf_scores.json; score_episodes_by_id()
    registry.py         VaeRegistry: publish/load checkpoint + reference_latents.npz
  stage_a_robometer/
    worker.py           Poll episode queue → score → write catalog → delete
    entrypoint.py       SIGTERM-safe entrypoint
  stage_b_deminf/
    accumulator.py      Sequential + parallel cohort accumulation
    train_job.py        Subprocess wrapper for scripts/train.py
    infer_worker.py     Poll cohort queue → score_episodes_by_id → write catalog
    entrypoint.py
  decision/
    decide.py           Pure function: Robometer gate → DemInf trim → keep/drop
    calibrate.py        Sweep thresholds against labeled validation set
    materialize.py      Copy kept episodes → clean bucket
  ingestion/
    handler.py          S3 PUT trigger → parse metadata → enqueue
  cli.py                rdf CLI

configs/
  quality/
    clean_data_vae.py   BetaVAE config (SA joint, z=18) for Franka clean_data episodes
  embodiments/          Per-robot MCAP topic names (franka.yaml, ...)
  thresholds/           Per-task pass thresholds
  pipeline.yaml         Mode (sequential/parallel), cohort sizing

scripts/
  preprocess_and_score.py    Decord preprocessing → Robometer scoring → DemInf symlinks
  run_with_deminf.sh         Full train pipeline wrapper
  deminf_score_episodes.py   One-shot DemInf scoring (openx env, checkpoint loaded once)
  run_local_pipeline.py      Full harness pipeline against /data/clean_data/
```

---

## Python environments

| Stage | Env | Location |
|---|---|---|
| Preprocessing, Stage A, harness | robometer uv | `/data/robometer/.venv/bin/python3` |
| DemInf training, scoring | openx conda | `/data/.conda/envs/openx/bin/python3` |

Both upstream repos are imported directly — do **not** reinstall or modify them.

---

## Model details

### Stage A — Robometer-4B

- **Model**: `robometer/Robometer-4B` (4B parameter VLM)
- **Server**: `eval_server.py` (FastAPI, port 8001) — loaded once, serves all workers
- **Input**: video frames at 1–2fps, task instruction string
- **Output**: `reward` (mean progress), `success_pred` (final frame success probability)
- **Threshold**: `success_pred ≥ 0.5`
- **Latency**: ~1.5s/ep from cached .npz frames, ~8–14s/ep when decoding inline

### Stage B — DemInf BetaVAE

- **Architecture**: Joint state+action BetaVAE, `z_dim=18`, `beta=0.05`
- **Encoder**: `MultiEncoder → Concatenate(flatten_time=True) → MLP([512,512])`
- **Input structure**:
  - `observation.state`: `{JOINT_POS(18): GAUSSIAN, MISC(14): GAUSSIAN}`
  - `action.desired_absolute`: `{JOINT_POS(16): GAUSSIAN, GRIPPER(2): BOUNDS, MISC(18): GAUSSIAN}`
- **Flat tensor order** (after `tf.nest.flatten`, alphabetical):
  - State (32-dim): `[JOINT_POS(18), MISC(14)]`
  - Action (36-dim): `[GRIPPER(2), JOINT_POS(16), MISC(18)]`
- **Scoring**: kSG mutual information estimator (`k=5,6,7`), normalized across the batch
- **Training data**: 24 episodes (train split), 6 episodes (test split) from Robometer-passing episodes
- **Training**: 1000 steps, batch=32, lr=1e-4, Adam

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `RDF_MODELS` | `mock` | `mock` or `real` |
| `RDF_STORAGE` | `local` | `local` or `s3` |
| `RDF_QUEUE` | `local` | `local` or `sqs` |
| `RDF_MAX_EPISODES` | all | Limit Stage A to N episodes (smoke-test mode) |
| `RDF_N_EPISODES` | `30` | Episodes to preprocess in preprocess_and_score.py |
| `RDF_INSTRUCTION` | `"pick the red cube and place in the blue box"` | Task instruction |
| `RDF_THRESHOLD` | `0.5` | Robometer success_pred threshold |
| `RDF_ROBOMETER_SERVER_URL` | `http://localhost:8001` | Robometer eval server |
| `RDF_DEMINF_CKPT` | auto-detected latest | Path to DemInf checkpoint step dir |
| `RDF_DEMINF_DATA` | `/tmp/rdf_pipeline_deminf/deminf_data` | DemInf data root |
| `RDF_DEMINF_SCORES` | `/tmp/rdf_deminf_scores.json` | Pre-computed per-episode scores |
| `RDF_DEMINF_THRESHOLD` | `-10.0` | DemInf score threshold (permissive default) |
| `RDF_DEMINF_SPLIT` | `train` | Split to score in deminf_score_episodes.py |

---

## Key design decisions

1. **Preprocessing separated from inference**: frames decoded once with `decord` and cached as `.npz`. Robometer server reads cached frames — ~6× faster than inline decode.

2. **Both models load once**: Robometer loads into GPU memory via `eval_server.py` (HTTP sidecar). DemInf checkpoint loaded once in `deminf_score_episodes.py` (JAX), results written to JSON. Stage B worker reads JSON — no JAX in the harness process.

3. **Cross-env boundary**: harness runs in robometer uv env; DemInf training and scoring run in openx conda env. Communication via JSON file (`rdf_deminf_scores.json`). No subprocess-per-episode.

4. **WandB bypass**: training uses a stub at `/tmp/wandb_stub/wandb/__init__.py` injected via `PYTHONPATH=/tmp/wandb_stub`. Full checkpoint saving still works (orbax, not WandB).

5. **Idempotent**: re-running any stage on already-scored episodes produces no duplicates (dedup on `model_version` / `vae_version`).
