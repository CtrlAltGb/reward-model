# Robot Data-Filtering Pipeline

A two-stage pipeline for filtering robot demonstration episodes before they enter training. Stage A gates on task completion (Robometer reward model). Stage B trims low-quality demonstrations (DemInf mutual-information score). Only episodes that pass both stages are written to the clean S3 bucket.

```
S3 raw bucket
    │
    ▼
[ Stage A — Robometer ]   persistent process, uv env at /data/robometer
    │  score_episode(video, instruction) → success_pred
    │  gate: success_pred ≥ robometer_threshold → pass / drop(task_incomplete)
    ▼
[ Stage B — DemInf ]      persistent process, conda env: openx
    │  encode(states, actions) → latents
    │  kNN-MI vs reference set → deminf_score
    │  trim: deminf_score ≥ deminf_threshold → pass / drop(low_quality_jitter)
    ▼
[ Decision ]              pure function, no model calls
    ▼
[ Materializer ]          copy kept episodes → S3 clean bucket
    ▼
S3 clean bucket (training input)
```

---

## Prerequisites

| Component | Env | Location |
|---|---|---|
| Robometer model | uv env | `/data/robometer` |
| DemInf VAEs | conda env `openx` | `/data/demonstration-information` |
| Pipeline harness | conda env `openx` or any Python ≥3.10 | this repo |

Both upstream repos must be present and their environments installed. Do not reinstall or modify them — the pipeline imports from them directly.

---

## Local setup

```bash
cd /data/reward_model
pip install -e ".[dev]"
```

Copy and fill in env vars:
```bash
cp .env.example .env
```

All backends default to **local/mock** — no AWS or GPU needed for development and testing.

---

## Running tests

```bash
# Unit + integration (mock models, local backends)
make test

# End-to-end (both sequential and parallel modes)
make e2e

# Full suite
make install && make lint && make test && make e2e
```

Expected output: **50 tests, all passing, no GPU or AWS required.**

---

## Running the pipeline (mock mode)

```bash
# Stage A worker
RDF_MODELS=mock rdf robometer-worker run --threshold 0.5

# Stage B worker (separate terminal)
RDF_MODELS=mock rdf deminf-worker run --threshold 0.0

# Decide + materialize a task
rdf decide pick_cup --robometer-threshold 0.5 --deminf-threshold 0.0
rdf materialize pick_cup
```

---

## Running with real models

### Stage A — start the Robometer eval server

The Robometer server loads the model once and serves all Stage A workers via HTTP. Start it inside the robometer uv env:

```bash
cd /data/robometer
uv run python robometer/evals/eval_server.py \
    model_path=robometer/Robometer-4B \
    batch_size=16 \
    num_gpus=1 \
    server_port=8001
```

Wait for `"Multi-GPU eval server initialized"` in the logs, then start Stage A:

```bash
RDF_MODELS=real \
RDF_ROBOMETER_SERVER_URL=http://localhost:8001 \
RDF_STORAGE=s3 \
RDF_QUEUE=sqs \
RDF_CATALOG=aws \
rdf robometer-worker run --threshold 0.5
```

Multiple Stage A replicas can share the same server.

### Stage B — run inside the openx conda env

```bash
conda activate openx
RDF_MODELS=real \
RDF_DEMINF_OBS_CKPT=/path/to/obs_vae \
RDF_DEMINF_ACTION_CKPT=/path/to/action_vae \
RDF_STORAGE=s3 \
RDF_QUEUE=sqs \
RDF_CATALOG=aws \
rdf deminf-worker run --threshold 0.0
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `RDF_MODELS` | `mock` | `mock` or `real` |
| `RDF_STORAGE` | `local` | `local` or `s3` |
| `RDF_QUEUE` | `local` | `local` or `sqs` |
| `RDF_CATALOG` | `local` | `local` or `aws` |
| `RDF_ROBOMETER_SERVER_URL` | `http://localhost:8001` | Robometer eval server URL |
| `RDF_ROBOMETER_MODEL_PATH` | `robometer/Robometer-4B` | Model path / HF hub ID |
| `RDF_DEMINF_OBS_CKPT` | — | Path to obs VAE orbax checkpoint |
| `RDF_DEMINF_ACTION_CKPT` | — | Path to action VAE orbax checkpoint |
| `RDF_S3_RAW_BUCKET` | `rdf-raw` | Raw episode bucket |
| `RDF_S3_CLEAN_BUCKET` | `rdf-clean` | Filtered output bucket |
| `RDF_SQS_EPISODE_QUEUE_URL` | — | SQS URL for episode queue |
| `RDF_SQS_COHORT_QUEUE_URL` | — | SQS URL for cohort queue |
| `RDF_LOCAL_STORAGE_PATH` | `/tmp/rdf/storage` | Local storage root |
| `RDF_LOCAL_QUEUE_PATH` | `/tmp/rdf/queues` | Local queue (SQLite) root |
| `RDF_LOCAL_CATALOG_PATH` | `/tmp/rdf/catalog` | Local catalog (SQLite) root |

---

## Pipeline modes

**Sequential** (default, for offline sweeps): Stage A processes all episodes first, then the accumulator emits one cohort per task to Stage B.

**Parallel** (for live ingestion): Stage B fires as soon as `deminf_cohort_min` (default: 50) passed episodes accumulate per task, or after `deminf_cohort_timeout_hours` (default: 12h), whichever comes first.

Set via `configs/pipeline.yaml` or the `PipelineConfig` schema.

---

## VAE registry

DemInf VAE checkpoints and pre-encoded reference latents are stored in a registry:

```
registry/task=<task>/vae_version=<v>/
    obs_vae/                  # orbax checkpoint
    action_vae/               # orbax checkpoint
    reference_latents.npz     # {obs_latents, action_latents} — encoded at train time
    meta.json
```

The reference latents are encoded once at train time (`stage_b_deminf/train_job.py`) and loaded by every inference worker at startup. This avoids re-encoding on every cohort.

---

## Repo layout

```
src/rdf/
  schemas/          Frozen Pydantic v2 models — the data contracts
  harness/          Storage, queue, catalog, config, logging, mcap, video
  models/           Mock + real model workers (robometer_worker, deminf_worker)
  stage_a_robometer/ Worker loop + entrypoint
  stage_b_deminf/   Accumulator, train job, infer worker + entrypoint
  decision/         decide.py, calibrate.py, materialize.py
  ingestion/        S3-triggered ingestion handler
  cli.py            rdf CLI
configs/
  embodiments/      Per-robot MCAP topic names
  thresholds/       Per-task pass/fail thresholds
  pipeline.yaml     Pipeline mode and cohort sizing
tests/
  test_schemas.py   Schema round-trips
  test_harness.py   Storage, queue, catalog backends
  test_models.py    Mock model — asserts model loads exactly once
  test_stage_a.py   Robometer worker loop
  test_decision.py  Decision logic + registry + materializer
  test_e2e.py       Full sequential and parallel pipeline
DECISIONS.md        Confirmed env state + all architectural decisions
```

---

## Key design constraints

- **Model loads once per worker process** — verified by `test_model_loaded_once` in both stage tests.
- **Robometer gate runs before DemInf** — enforced in `decide.py` and tested in `test_robometer_gate_before_deminf`.
- **Reference latents saved at train time** — `train_job.py` encodes the reference set and persists `reference_latents.npz` so inference workers load them without re-encoding.
- **Idempotent** — re-running any stage on already-scored episodes produces no duplicates (dedup on `model_version` / `vae_version`).
- **No secrets in repo** — env vars only; see `.env.example`.
