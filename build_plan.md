# BUILD PLAN — Robot Data-Filtering Pipeline

**Audience:** Claude Code (agentic build). Follow phase by phase. Run the verification step at the end of each phase before starting the next.

---

## 0. Operating rules (read first)

1. **No live AWS required.** All tests run against mock backends. Real AWS sits behind env-var flags.
2. **Robometer is already set up at `/data/robometer`.** Its uv env is fully installed. Do not reinstall or modify it.
3. **DemInf fork is already set up at `/data/demonstration-information`.** Its conda env `openx` is fully installed. Do not reinstall or modify it.
4. **Do not write env YAML files or specify any dependencies manually.** Instead:
   - For Robometer: read `/data/robometer/pyproject.toml` and `uv.lock` to understand what is available, then use as-is.
   - For DemInf: read `/data/demonstration-information/requirements.txt` and `DEPENDENCIES.md`, then run `conda run -n openx pip list` to confirm what is installed.
5. **Copy and adapt upstream scripts — do not subprocess into them per-episode.** Both upstream repos have model loading that must happen once. The pipeline writes its own long-running worker scripts that import from the upstream packages directly. See §7.3 and §7.4 for exactly what to read, copy, and rewrite.
6. **Schemas are contracts.** Pydantic models in Phase 1 are frozen. Every component reads/writes them.
7. **Test each phase before moving on.**
8. **Unknown details → `# DECISION-NEEDED:` comment + entry in `DECISIONS.md`.**
9. **Commit per phase.** Conventional commits.
10. **No secrets in the repo.** Env vars only. `.env.example` provided.

---

## 1. First thing Claude Code must do — read the existing envs and scripts

Before writing any code, read and confirm everything.

### Read Robometer
```bash
cat /data/robometer/pyproject.toml
cat /data/robometer/uv.lock                          # exact installed versions
cat /data/robometer/robometer/evals/eval_server.py   # read the HTTP server impl
cat /data/robometer/scripts/example_inference.py     # read the HTTP client impl
cat /data/robometer/scripts/example_inference_local.py  # read the local impl
```

From reading these, understand:
- How the model is loaded (what class, what config, what checkpoint arg)
- What `eval_server.py` exposes (endpoints, request/response format)
- What `example_inference.py` sends and receives
- What `example_inference_local.py` does differently (no server)

### Read DemInf
```bash
cat /data/demonstration-information/requirements.txt
cat /data/demonstration-information/DEPENDENCIES.md
conda run -n openx pip list                          # exact installed versions
cat /data/demonstration-information/scripts/quality/estimate_quality.py  # read the scoring script
cat /data/demonstration-information/scripts/train.py                     # read the training script
ls /data/demonstration-information/scripts/quality/                      # see what else is there
ls /data/demonstration-information/openx/                                # understand the package structure
```

From reading these, understand:
- How `estimate_quality.py` loads the VAE checkpoints
- What the scoring loop looks like (dataset iteration, encode, kNN-MI)
- What the output format is (how scores are written to disk)
- Whether it loads once or once-per-call

### Record what you find
Write confirmed Python versions, key package versions (torch, jax, transformers, flax), and the request/response format of the Robometer server into `DECISIONS.md` under "Confirmed env state". Do not proceed past Phase 0 until both envs pass their verification checks:

```bash
# Robometer
cd /data/robometer && uv run python -c "import robometer; print('OK')"

# DemInf
conda run -n openx python -c "import jax; print(jax.__version__, jax.devices())"
conda run -n openx python -c "import mcap; import av; import orbax.checkpoint; print('OK')"
```

---

## 2. The model-loading problem and the fix

### Why subprocess-per-episode is wrong

Both upstream scripts reload the model from disk on every invocation:
- `example_inference_local.py` — loads the full 4B checkpoint on every call. On an H100 this is ~15–30 seconds per episode just in model loading. Completely unacceptable for a pipeline processing thousands of episodes.
- `estimate_quality.py` — loads the VAE checkpoints and rebuilds the dataset iterator on every call. Same problem.

### The fix: persistent workers that import directly

The pipeline writes two new worker scripts inside `src/rdf/`. These import from the upstream packages (which are installed as editable packages into their respective envs), load the model once at startup, then process episodes in a loop from the queue. No subprocess. No per-episode model loading.

```
Stage A worker process (uv env at /data/robometer):
  startup:  load Robometer checkpoint once → model in GPU memory
  loop:     poll SQS → run inference → write result → next

Stage B worker process (conda env: openx):
  startup:  load frozen VAE checkpoints once → model in GPU memory
  loop:     poll cohort queue → encode batch → kNN-MI → write scores → next
```

For Robometer there is also a **server mode** already built into the repo (`eval_server.py` + `example_inference.py`). Claude Code should read `eval_server.py` first and decide whether to:
- **Option A (preferred if server is clean):** Use `eval_server.py` as-is to run the server, and write a thin HTTP client as the adapter. Model loads once in the server process. Multiple worker replicas can hit the same server.
- **Option B (if server doesn't fit the pipeline's request format):** Copy the model loading and inference logic from `example_inference_local.py` into `src/rdf/stage_a_robometer/robometer_worker.py`, which runs as a persistent process.

**Claude Code must read both scripts first and pick the option that requires less rewriting.** Document the choice in `DECISIONS.md`.

---

## 3. Pipeline overview

```
S3 raw bucket
    │  pull: mcap + head mp4 + metadata.yaml per episode
    ▼
[ Stage A — Robometer worker ]  (persistent process, uv env at /data/robometer)
    │  model loaded once at startup
    │  loop: poll episode queue → score → write to catalog → next
    │  gate:  success_pred ≥ robometer_threshold?
    │         fail → Drop (reason: task_incomplete)
    │         pass ↓
[ per-task accumulator ]
    │  sequential mode: wait for all of Stage A to finish
    │  parallel mode:   fire when cohort_min reached or timeout
    ▼
[ Stage B — DemInf worker ]  (persistent process, conda env: openx)
    │
    │  train (once per task, then frozen):
    │    VAEs loaded, trained on reference set, checkpointed
    │
    │  infer loop: poll cohort queue → load frozen VAEs once
    │    → encode batch → kNN-MI vs reference latents → write scores → next
    │
    │  trim: deminf_score ≥ deminf_threshold?
    │        fail → Drop (reason: low_quality_jitter)
    │        pass ↓
[ Decision step ]
    │  pure function over catalog scores, no model calls
    │  write final_decision + reasons → catalog
    ▼
[ Materializer ]
    │  copy kept episodes → S3 clean bucket
    ▼
S3 clean bucket (training input)
```

---

## 4. Repo layout

```
robot-data-filtering/
  pyproject.toml              # harness deps only (no torch/jax)
  .pre-commit-config.yaml
  .github/workflows/ci.yml
  .env.example
  README.md
  DECISIONS.md
  Makefile
  docker-compose.yml
  src/rdf/
    __init__.py
    schemas/
    harness/
    models/
      base.py                 # Protocol interfaces
      mock.py                 # deterministic mock implementations
      robometer_worker.py     # persistent worker — written by reading /data/robometer scripts
      deminf_worker.py        # persistent worker — written by reading /data/demonstration-information scripts
    stage_a_robometer/
      worker.py               # SQS poll loop, calls robometer_worker.py model
      entrypoint.py
      Dockerfile
    stage_b_deminf/
      accumulator.py
      train_job.py
      infer_worker.py         # SQS poll loop, calls deminf_worker.py model
      entrypoint.py
      Dockerfile
    decision/
      decide.py
      calibrate.py
      materialize.py
    ingestion/
      handler.py
    cli.py
  configs/
    embodiments/
    thresholds/
    pipeline.yaml
    deminf/
  infra/
    terraform/
    stepfunctions/
  tests/
    conftest.py
    fixtures/
```

---

## 5. Phase 1 — Schemas

`src/rdf/schemas/` — frozen pydantic v2 models.

**`EpisodeManifest`**
```
episode_id: str
s3_prefix: str
robot_id: str
embodiment: str
task: str
instruction: str
head_video_key: str
mcap_key: str
metadata_key: str
created_at: datetime
schema_version: str = "1"
```

**`RobometerResult`**
```
episode_id, task, embodiment: str
robometer_reward: float
robometer_success_pred: float
frames_used: int
model_version: str
latency_ms: float
scored_at: datetime
status: Literal["scored", "failed"]
error: str | None = None
```

**`CohortMessage`**
```
cohort_id: str
task: str
episode_ids: list[str]
vae_version: str
reference_set_version: str
created_at: datetime
```

**`DeminfResult`**
```
episode_id, task: str
deminf_score: float
vae_version: str
reference_set_version: str
cohort_id: str
scored_at: datetime
status: Literal["scored", "failed"]
error: str | None = None
```

**`CatalogRow`**
```
episode_id, task, embodiment, robot_id: str
robometer_reward: float | None
robometer_success_pred: float | None
robometer_pass: bool | None
deminf_score: float | None
deminf_pass: bool | None
final_decision: Literal["keep", "drop", "pending"] = "pending"
reasons: list[str] = []
robometer_model_version: str | None
vae_version: str | None
pipeline_mode: Literal["sequential", "parallel"]
created_at, updated_at: datetime
```

**Config models:**
- `EmbodimentConfig`: `name`, `head_camera`, `instruction_field`, `state_topics`, `action_topics`, `action_chunk_size`. **# DECISION-NEEDED: real mcap topic names**
- `ThresholdConfig`: `task`, `robometer_threshold`, `deminf_threshold`, `calibrated_against`, `vae_version`
- `PipelineConfig`: `mode`, bucket names, queue names, `deminf_cohort_min=50`, `deminf_cohort_max=300`, `deminf_cohort_timeout_hours=12.0`

**Verify:** every model round-trips to/from JSON and YAML.

---

## 6. Phase 2 — Harness library

`src/rdf/harness/` — shared by both stages. Local/mock backends by default.

- **`logging.py`** — structlog JSON; `bind_episode` / `bind_cohort` context helpers.
- **`config.py`** — load/validate `configs/*`; cached accessors.
- **`storage.py`** — abstract `ObjectStore`. `LocalObjectStore` + `S3ObjectStore`. `RDF_STORAGE=local|s3`.
- **`queue.py`** — abstract `WorkQueue`. `LocalQueue` (sqlite) + `SqsQueue`. `RDF_QUEUE=local|sqs`.
- **`catalog.py`** — abstract `Catalog`. `LocalCatalog` (parquet + sqlite) + `AwsCatalog` (S3 + DynamoDB). Keyed by `episode_id`; re-processing overwrites.
- **`mcap_extract.py`** — `extract_state_action(mcap_bytes, EmbodimentConfig)`. `McapReader` interface + `SyntheticMcapReader` for tests. **# DECISION-NEEDED: real mcap encoding**
- **`video.py`** — `extract_frames(mp4_bytes, n_frames)`. Stub returns dummy arrays when av not importable.
- **`idempotency.py`** — skip if already scored at this `model_version`.

**Verify:** moto-backed tests for all backends.

---

## 7. Phase 3 — Model workers (the critical phase)

This is where Claude Code reads the upstream scripts and writes the persistent workers. Do not skip the reading step.

### 7.1 Interfaces (`src/rdf/models/base.py`)

```python
class RobometerModel(Protocol):
    """Loaded once, called many times."""
    def score_episode(self, video_path: str, instruction: str) -> RobometerScore: ...

class DeminfModel(Protocol):
    """Loaded once, called many times."""
    def encode_episodes(self, episode_states: list[np.ndarray],
                        episode_actions: list[np.ndarray]) -> np.ndarray: ...
    def score_against_reference(self, episode_latents: np.ndarray,
                                reference_latents: np.ndarray,
                                k: int = 5) -> list[float]: ...
```

### 7.2 Mock (`src/rdf/models/mock.py`, `RDF_MODELS=mock`)

Deterministic, seedable. Default in all tests. No torch/jax imports.

### 7.3 Robometer persistent worker (`src/rdf/models/robometer_worker.py`)

**Step 1 — Read these files before writing anything:**
```bash
cat /data/robometer/robometer/evals/eval_server.py
cat /data/robometer/scripts/example_inference_local.py
cat /data/robometer/scripts/example_inference.py
```

**Step 2 — Decide the approach based on what you read:**

If `eval_server.py` exposes a clean HTTP endpoint that accepts `{video_path, task}` and returns `{reward, success_pred}` (or equivalent JSON), use **server mode**:
```
# Server mode (Option A):
# eval_server.py runs as a sidecar process, loads model once
# robometer_worker.py is a thin HTTP client
# Multiple worker replicas share one server → very efficient
```

If the server interface is more complex or the response format needs adaptation, use **direct import mode**:
```
# Direct import mode (Option B):
# Copy the model loading logic from example_inference_local.py
# into robometer_worker.py as a class that loads once in __init__
# and exposes score_episode() for the queue loop
```

**Step 3 — Write the worker.** Whichever option you choose, the pattern is:

```python
# src/rdf/models/robometer_worker.py
# REAL-MODEL SEAM
# Written by reading /data/robometer/scripts/example_inference_local.py
# and /data/robometer/robometer/evals/eval_server.py
#
# This file must be run inside the robometer uv env:
#   cd /data/robometer && uv run python -m rdf.models.robometer_worker
#
# DO NOT modify /data/robometer — only import from it.

class RobometerWorker:
    def __init__(self, model_path: str):
        # Load checkpoint ONCE here — copy loading logic from example_inference_local.py
        # or connect to the already-running eval_server
        ...

    def score_episode(self, video_path: str, instruction: str) -> RobometerScore:
        # Run inference on already-loaded model
        # Copy inference logic from example_inference_local.py
        # Return RobometerScore (our schema, not upstream's)
        ...
```

**Step 4 — Verify it starts and scores one episode without reloading:**
```bash
cd /data/robometer
uv run python -c "
from rdf.models.robometer_worker import RobometerWorker
w = RobometerWorker(model_path='robometer/Robometer-4B')
r1 = w.score_episode('/tmp/test.mp4', 'pick up the cup')
r2 = w.score_episode('/tmp/test.mp4', 'pick up the cup')
print('scored twice, model loaded once:', r1, r2)
"
```

### 7.4 DemInf persistent worker (`src/rdf/models/deminf_worker.py`)

**Step 1 — Read these files before writing anything:**
```bash
cat /data/demonstration-information/scripts/quality/estimate_quality.py
ls /data/demonstration-information/scripts/quality/
cat /data/demonstration-information/openx/data/    # understand data pipeline
# Read whichever quality_estimators file is referenced by estimate_quality.py
```

**Step 2 — Understand the loading pattern in `estimate_quality.py`:**

The script almost certainly: loads VAE checkpoints → builds a dataset → encodes all episodes → runs kNN-MI. The problem is it does all of this in one shot per invocation. For the pipeline we need to separate these into:
- `load_vaes(obs_ckpt, action_ckpt)` — called once at startup or once per new VAE version
- `encode_batch(episodes)` — called per cohort
- `score_against_reference(cohort_latents, reference_latents)` — called per cohort

**Step 3 — Write the worker.** Copy the relevant loading and scoring logic from `estimate_quality.py` and the quality estimator classes it references:

```python
# src/rdf/models/deminf_worker.py
# REAL-MODEL SEAM
# Written by reading:
#   /data/demonstration-information/scripts/quality/estimate_quality.py
#   /data/demonstration-information/scripts/quality/quality_estimators.py (or equivalent)
#
# This file must be run inside the openx conda env:
#   conda run -n openx python -m rdf.models.deminf_worker
#
# DO NOT modify /data/demonstration-information — only import from it.

class DeminfWorker:
    def __init__(self, obs_ckpt: str, reference_latents_path: str):
        # Load frozen VAE checkpoints ONCE here
        # Copy checkpoint loading logic from estimate_quality.py
        # Load reference latents from registry
        ...

    def score_cohort(self, episode_states: list[np.ndarray],
                     episode_actions: list[np.ndarray],
                     k: int = 5) -> list[float]:
        # Encode with frozen VAEs, compute kNN-MI vs reference latents
        # Copy encoding + MI estimation logic from estimate_quality.py
        # Return per-episode scores
        ...
```

**Step 4 — Verify:**
```bash
conda run -n openx python -c "
from rdf.models.deminf_worker import DeminfWorker
w = DeminfWorker(obs_ckpt='...', reference_latents_path='...')
scores = w.score_cohort([states1, states2], [actions1, actions2])
print('scored cohort:', scores)
"
```

### 7.5 VAE Registry (`src/rdf/models/registry.py`)

```
registry/task=<task>/vae_version=<v>/
    obs_vae/                  # checkpoint dir from scripts/train.py
    reference_latents.npz     # encoded reference set, saved at train time
    meta.json
```

Methods: `publish(task, artifact)`, `load(task, version="current")`, `current_version(task)`.

**Verify:** mock score calls return valid results; registry publish → load round-trips.

---

## 8. Phase 4 — Stage A: Robometer worker loop

`src/rdf/stage_a_robometer/worker.py`:

```python
# Startup: instantiate RobometerWorker (loads model once)
# Loop:
#   1. poll episode queue
#   2. idempotency check
#   3. storage.get_bytes(head_video_key) → write to tmp file
#   4. worker.score_episode(tmp_path, instruction)
#   5. write RobometerResult to catalog
#   6. delete message; on failure → DLQ
```

`entrypoint.py` + CLI `rdf robometer-worker run` — graceful SIGTERM.

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
# /data/robometer mounted at runtime (contains the uv env and checkpoint)
COPY src/rdf /app/rdf
WORKDIR /data/robometer
RUN uv pip install -e /app/rdf[harness]
ENTRYPOINT ["uv", "run", "python", "-m", "rdf.stage_a_robometer.entrypoint"]
```

**Verify:** mock models + local queue/store/catalog: N episodes → queue drains → N scored rows, 0 duplicates, poison episode in DLQ.

---

## 9. Phase 5 — Stage B: DemInf train + infer

### 9.1 Training (`train_job.py`)

Calls `scripts/train.py` via subprocess (training is a one-off job, not a hot loop — subprocess is fine here):
```bash
conda run -n openx python \
  /data/demonstration-information/scripts/train.py \
  --config /data/demonstration-information/configs/bc/manav.py:default \
  --path <registry_dir>/<task>/<vae_version>/ \
  --name <task>_vae
```

After training completes, encode the reference set using `DeminfWorker` and save `reference_latents.npz` to the registry alongside the checkpoint. This is the one extra step the upstream script doesn't do — the reference latents must be persisted so the infer worker can load them without re-encoding.

### 9.2 Inference worker loop (`infer_worker.py`)

```python
# Startup: load DeminfWorker with current VAE ckpt + reference latents (once)
# Loop:
#   1. poll cohort queue
#   2. load episode (states, actions) via mcap_extract for each episode_id in cohort
#   3. worker.score_cohort(states_list, actions_list)
#   4. write DeminfResult rows to catalog
#   5. delete cohort message; checkpoint progress
```

### 9.3 Accumulator (`accumulator.py`)

- `sequential`: waits for `stage_a_complete` sentinel, emits one cohort per task.
- `parallel`: fires when `cohort_min` passed episodes accumulate OR timeout elapses.

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install -y wget git patchelf && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/mc.sh && bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh
ENV PATH="/opt/conda/bin:$PATH"
# /data/demonstration-information mounted at runtime (contains conda env and checkpoints)
COPY src/rdf /app/rdf
RUN conda run -n openx pip install -e /app/rdf[harness]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "openx", \
            "python", "-m", "rdf.stage_b_deminf.entrypoint"]
```

**Verify:** train on synthetic reference set → save reference latents → infer cohort with model loaded once → all episodes scored; restart resumes without rescoring.

---

## 10. Phase 6 — Decision, calibration, materializer

`src/rdf/decision/`:

- **`decide.py`** — pure function, no model calls:
  1. Robometer gate: fail → `drop` + `task_incomplete`
  2. DemInf trim: fail → `drop` + `low_quality_jitter`
  3. Both pass → `keep`

- **`calibrate.py`** — sweep thresholds against labeled validation set → `configs/thresholds/<task>.yaml`. **# DECISION-NEEDED: validation set format**

- **`materialize.py`** — copy `keep` episodes → clean bucket. Idempotent. **# DECISION-NEEDED: output layout**

---

## 11. Phase 7 — Ingestion

`src/rdf/ingestion/handler.py` — triggered by S3 PUT on `*.mp4`. Check all three files exist → parse `metadata.yaml` → build `EpisodeManifest` → enqueue idempotently on `recording_id`.

---

## 12. Phase 8 — Infra

- **`docker-compose.yml`** — `robometer-worker`, `deminf-runner`, optional `localstack`. Both mount `/data/` so existing repos and checkpoints are accessible. GPU passthrough via `deploy.resources.reservations.devices`.
- **`infra/terraform/`** — S3 buckets, SQS queues + DLQs, DynamoDB, Lambda, AWS Batch job defs, Glue/Athena, IAM.
- **`infra/stepfunctions/deminf_pipeline.json`** — `accumulate → check_vae_stale → (train if stale) → infer → decision`.

---

## 13. Phase 9 — Observability

Metric emission: queue depth, episodes/min, per-stage latency, per-task pass/fail rates, DLQ size, GPU util. CloudWatch dashboard + alarm stubs.

---

## 14. Phase 10 — End-to-end + docs

- `tests/fixtures/generate_episode.py` — synthetic good/bad episodes.
- `make e2e` — full chain in mock mode, both sequential and parallel modes.
- `README.md` — local setup, how to run the server (Robometer option A), how to switch modes.

**Acceptance criteria:**
- [ ] `make install && make lint && make test && make e2e` green, no GPU/AWS needed
- [ ] `RDF_MODELS=real` changes only `robometer_worker.py` and `deminf_worker.py`
- [ ] Model loads once per worker process — verified by test that counts model init calls
- [ ] Robometer gate before DemInf — enforced and tested
- [ ] DemInf `reference_latents.npz` saved at train time, loaded at infer time — tested
- [ ] Idempotency: re-running any stage produces no duplicates
- [ ] Both sequential and parallel modes tested in e2e
- [ ] `DECISIONS.md` has confirmed env state + all items

---

## 15. Known unknowns — DECISIONS.md items

| # | Item | Default |
|---|---|---|
| 1 | Confirmed env state | Fill in during Phase 0 |
| 2 | Robometer server vs direct import | Read `eval_server.py` and decide; document choice |
| 3 | Robometer response format | Read `example_inference.py` to get exact field names |
| 4 | DemInf VAE loading API | Read `estimate_quality.py` to get exact class/method names |
| 5 | mcap topic names per embodiment | Synthetic in tests; real names in `configs/embodiments/` |
| 6 | DemInf task config | Template from `configs/bc/manav.py`; fill dataset path |
| 7 | Robometer model checkpoint | `robometer/Robometer-4B`; confirm if fine-tuned checkpoint used |
| 8 | Training pipeline output format | Flat copy `clean/task=<t>/episode=<id>/` |
| 9 | Pipeline mode default | `sequential` for sweeps, `parallel` for live ingestion |
| 10 | DemInf cohort trigger | `cohort_min=50`, `cohort_max=300`, `timeout=12h` |

---

## 16. Commit sequence

```
chore: bootstrap repo (after Phase 0 env checks pass)
feat: schemas (Phase 1)
feat: harness backends + mocks (Phase 2)
feat: model workers — read upstream scripts, write persistent loaders (Phase 3)
feat: robometer worker loop (Phase 4)
feat: deminf train + infer worker loop (Phase 5)
feat: decision + calibrate + materializer (Phase 6)
feat: ingestion handler (Phase 7)
feat: infra + compose (Phase 8)
feat: observability (Phase 9)
test: e2e + docs (Phase 10)
```

**Hard stop after Phase 0.** Report confirmed env state. Fix any failures before proceeding.

**Hard stop after Phase 3.** Report:
1. Which Robometer option was chosen (server vs direct import) and why
2. Exact field names of the Robometer inference output
3. Exact class/method used for DemInf VAE loading from `estimate_quality.py`
4. Whether `reference_latents.npz` saving needs to be added to the train job

These four answers determine the correctness of every stage that follows.