# Robot Data-Filtering Pipeline (RDF)

Two-stage pipeline for filtering robot demonstration episodes before training.
**Stage A** gates on task completion (Robometer VLM). **Stage B** trims low-quality
demonstrations (DemInf mutual-information VAE). Only episodes passing both stages
are kept.

---

## Pipeline overview

```
/data/clean_data/          raw episodes (MCAP + MP4 + metadata.yaml)
        │
        ▼
[ Ingestion ]              build EpisodeManifests → seed SQLite catalog → enqueue
        │
        ▼
[ Stage A — Robometer ]    RobometerLocalWorker (in-process, ~70s load)
  streaming producer/consumer: 8 decode workers → Queue(maxsize=8) → GPU scorer
  success_pred ≥ robometer_threshold (0.5) → pass / drop(task_incomplete)
        │  pass ↓
        ▼
[ DemInf Scoring ]         deminf_score_episodes.py  (openx conda env, subprocess)
  loads VAE checkpoint once → kSG MI score per episode → rdf_deminf_scores.json
        │
        ▼
[ Cohort Accumulation ]    accumulate_sequential: one cohort per task
        │
        ▼
[ Stage B — DemInf ]       DeminfWorker: JSON lookup table, no JAX in harness
  deminf_score ≥ deminf_threshold (-10.0) → pass / drop(low_quality_jitter)
        │
        ▼
[ Decision ]               decide_all(): pure function → keep / drop / pending
        ▼
SQLite catalog  /data/reward_model_files/rdf_integration/catalog/catalog.db
```

**Verified on 177 episodes**: 176 keep (99.4%), 1 drop, 0 pending, ~188s total.

---

## Installation

### Prerequisites

- NVIDIA GPU (tested on A10G, 22 GB VRAM)
- [uv](https://docs.astral.sh/uv/) for the robometer environment
- [conda](https://docs.conda.io/) for the openx environment

### 1 — Clone with submodules

```bash
git clone --recurse-submodules <repo-url> /data/reward_model
cd /data/reward_model
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2 — Set up the robometer uv environment

```bash
cd /data/reward_model/robometer
uv sync
```

Or from the pinned freeze (exact versions used in production):

```bash
cd /data/reward_model/robometer
uv venv
uv pip install -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple
```

> `torch==2.8.0+cu128` and `torchvision==0.23.0+cu128` require the PyTorch CUDA index.

### 3 — Set up the openx conda environment

```bash
conda env create -f /data/reward_model/demonstration-information/environment.yaml
```

This creates the `openx` environment with Python 3.11 and all pinned dependencies
(JAX, TensorFlow, Flax, MCAP, etc.).

Verify:

```bash
/data/.conda/envs/openx/bin/python3 -c "import jax; print(jax.__version__)"
```

### 4 — Install the rdf package (openx env)

```bash
/data/.conda/envs/openx/bin/pip install -e /data/reward_model
```

---

## Configuration

All paths and tuning knobs live in `configs/` — edit these before running:

| File | Controls |
|------|----------|
| `configs/paths.yaml` | All filesystem paths: data dir, scratch, checkpoints, Python envs, upstream repos |
| `configs/pipeline.yaml` | Thresholds (`robometer_threshold`, `deminf_threshold`), mode, versioning, embodiment |
| `configs/models.yaml` | Model version, video FPS/size/workers, DemInf estimator/batch/split |

Key paths to verify before first run:

```yaml
# configs/paths.yaml
clean_data_dir: /data/clean_data          # raw episodes
scratch_dir: /data/reward_model_files/rdf_integration
deminf_ckpts_dir: /data/reward_model_files/rdf_deminf_ckpts
openx_python: /data/.conda/envs/openx/bin/python3
robometer_root: /data/robometer
robometer_model_path: /data/robometer/robometer/Robometer-4B
```

Env vars override any YAML value at runtime:

```bash
RDF_ROBOMETER_THRESHOLD=0.7   # override robometer gate
RDF_DEMINF_THRESHOLD=-5.0     # override DemInf trim
RDF_DEMINF_DATA=/path/to/data # override deminf data dir
RDF_MAX_EPISODES=30           # smoke-test: process only first N episodes
RDF_INSTRUCTION="pick cube"   # override instruction for all episodes
```

---

## Running the pipeline

### Full run (all episodes)

```bash
cd /data/reward_model
/data/robometer/.venv/bin/python3 -u scripts/run_local_pipeline.py
```

### Smoke test (first 30 episodes)

```bash
RDF_MAX_EPISODES=30 /data/robometer/.venv/bin/python3 -u scripts/run_local_pipeline.py
```

### Reset and re-run from scratch

```bash
rm -rf /data/reward_model_files/rdf_integration
/data/robometer/.venv/bin/python3 -u scripts/run_local_pipeline.py
```

### Run tests (openx env, no GPU needed)

```bash
/data/.conda/envs/openx/bin/python3 -m pytest tests/ -v --tb=short
```

---

## Two-environment architecture

The pipeline deliberately spans two Python environments that cannot be merged:

| Env | Location | Used for |
|-----|----------|---------|
| robometer uv | `/data/robometer/.venv/bin/python3` | Harness, Stage A (torch, decord) |
| openx conda | `/data/.conda/envs/openx/bin/python3` | DemInf scoring (JAX, Flax, TF, MCAP); tests |

The two envs communicate via a single JSON file (`rdf_deminf_scores.json`).
DemInf scoring runs once as a subprocess; the harness reads scores as a lookup table.

Both upstream repos are imported by `sys.path` injection — do **not** install or modify them:
- `/data/robometer` → Robometer VLM
- `/data/demonstration-information` → DemInf quality estimators

---

## Source layout

```
src/rdf/
  schemas/models.py           All frozen Pydantic v2 schemas — change here first
  harness/
    catalog.py                LocalCatalog (SQLite) + AwsCatalog (DynamoDB)
    queue.py                  LocalQueue (SQLite) + SqsQueue
    storage.py                ObjectStore: Local + S3
    video.py                  MP4 decode → center-crop → 256×256 @ 2fps
    config.py                 YAML config loaders (get_paths/models/pipeline_config)
    idempotency.py            Skip already-scored episodes
    mcap_extract.py           MCAP → (states, actions)
    logging.py                structlog JSON logging
  models/
    robometer_worker.py       RobometerLocalWorker (in-process) + RobometerWorker (HTTP)
    deminf_worker.py          JSON score lookup table for Stage B
    registry.py               VaeRegistry: publish/load checkpoint artifacts
    base.py                   Protocol interfaces
  stage_a_robometer/
    worker.py                 stream_episodes() + run_worker(): streaming producer/consumer
  stage_b_deminf/
    accumulator.py            Sequential + parallel cohort accumulation
    infer_worker.py           Cohort queue → score_episodes_by_id → catalog
    train_job.py              Subprocess wrapper for VAE training
  decision/
    decide.py                 Pure function: gate + trim → keep/drop
    calibrate.py              Sweep thresholds against labeled validation set
    materialize.py            Copy kept episodes to clean bucket

configs/
  paths.yaml                  All filesystem paths
  pipeline.yaml               Thresholds, mode, cohort sizing, versioning
  models.yaml                 Model version, video settings, DemInf estimator
  quality/clean_data_vae.py   BetaVAE config (SA joint, z=18) for Franka episodes

scripts/
  run_local_pipeline.py       Full end-to-end pipeline runner
  deminf_score_episodes.py    Standalone DemInf scorer (openx env)
```

---

## Model details

### Stage A — Robometer-4B

| | |
|---|---|
| Architecture | 4B parameter VLM (Qwen3-VL based) |
| Loading | In-process via `load_model_from_hf`, loaded once (~70s) |
| Input | 8 evenly-spaced frames (subsampled from 2fps decode), task instruction |
| Output | `success_pred` (final-frame success probability), `reward` (mean progress) |
| Threshold | `success_pred ≥ 0.5` (configurable in `configs/pipeline.yaml`) |
| Throughput | ~530ms/episode on A10G |

### Stage B — DemInf BetaVAE

| | |
|---|---|
| Architecture | Joint state+action BetaVAE, `z_dim=18`, `beta=0.05`, MLP encoder/decoder |
| Scoring | kSG mutual-information estimator in joint latent space |
| Input | Proprioceptive state + action sequences from MCAP (no images) |
| Active checkpoint | `/data/reward_model_files/rdf_deminf_ckpts/clean_data_vae_20260626_181842/1000` |
| Throughput | ~130ms/episode (cached `.npz`), ~3s/episode (fresh MCAP parse + JAX JIT) |

---

## Pipeline data locations

```
/data/clean_data/                          raw input episodes
/data/reward_model_files/
  rdf_integration/
    catalog/catalog.db                     SQLite catalog (CatalogRow per episode)
    queues/                                SQLite episode + cohort queues
  rdf_pipeline_deminf/deminf_data/train/  episode symlinks + _cached.npz files
  rdf_deminf_scores.json                  per-episode DemInf scores (written each run)
  rdf_deminf_ckpts/                        pre-trained VAE checkpoints
```
