# DECISIONS.md — Robot Data-Filtering Pipeline

---

## 1. Confirmed env state (Phase 0)

### Robometer (uv env at /data/robometer)
- **Python**: 3.10 (pinned: `requires-python = "==3.10.*"`)
- **torch**: 2.8.0 (CUDA 12.8 build)
- **transformers**: ≥4.57
- **trl**: 0.20.0
- **fastapi**: ≥0.116.1, **uvicorn**: ≥0.35.0
- **Model class**: Loaded via `robometer.utils.save.load_model_from_hf(model_path, device)` → returns `(exp_config, tokenizer, processor, reward_model)`
- **Server**: FastAPI multi-GPU server in `robometer/evals/eval_server.py`
- **Verification**: `uv run python -c "import robometer; print('OK')"` → **OK**

### DemInf (conda env: openx, Python 3.11)
- **Python**: 3.11
- **jax**: 0.4.37, **jaxlib**: 0.4.36 (CUDA 12 build — one patch behind jax, expected)
- **flax**: 0.10.2
- **optax**: 0.2.4
- **orbax-checkpoint**: 0.11.5
- **numpy**: 1.26.4 (pinned <2.0)
- **mcap**: 1.4.0, **av**: 17.1.0, **pyarrow**: 24.0.0
- **Checkpoint loader**: `openx.utils.evaluate.load_checkpoint(path)` → `(alg, state, dataset_statistics, config)`
- **Encoding**: `alg.predict(state, batch, rng)` → JAX array of latent embeddings
- **Verification**: `conda run -n openx python -c "import jax; print(jax.__version__, jax.devices())"` → **0.4.37 [CudaDevice(id=0)]**
- **Verification**: `conda run -n openx python -c "import mcap; import av; import orbax.checkpoint; print('OK')"` → **OK**

---

## 2. Robometer: server vs direct import

**Choice: Option A — server mode**

Rationale:
- `eval_server.py` is a clean, complete FastAPI multi-GPU server. Loads model once, serves via `/evaluate_batch_npy`.
- `example_inference.py` already provides `compute_rewards_per_frame(eval_server_url, video_frames, task)` → `(progress_array, success_array)`.
- Zero rewriting needed. Our `robometer_worker.py` is a thin HTTP client.
- Multiple Stage A worker replicas can share one server process → very efficient.

Server endpoint used: `POST /evaluate_batch_npy` (multipart, accepts .npy blobs).

---

## 3. Robometer response format (exact field names)

From `example_inference.py::extract_rewards_from_server_output()` and `compute_rewards_per_frame()`:

```
Server response JSON:
{
  "outputs_progress": {
    "progress_pred": [[0.1, 0.3, 0.5, 0.8, 0.9], ...]   # list[list[float]], one per sample
  },
  "outputs_success": {
    "success_probs": [[0.2, 0.4, 0.7, 0.9, 0.95], ...]  # list[list[float]], one per sample
  },
  "outputs_preference": {...}   # not used in our pipeline
}
```

Our pipeline maps:
- `robometer_reward` = mean of `progress_pred[0]` (or last value — mean preferred for stability)
- `robometer_success_pred` = last value of `success_probs[0]` (final-frame success probability)

---

## 4. DemInf VAE loading (exact class/method names)

From `quality_estimators.py::get_dataset_and_score_fn()` and `openx/utils/evaluate.py::load_checkpoint()`:

```python
from openx.utils.evaluate import load_checkpoint

obs_alg, obs_state, obs_ds_stats, obs_config = load_checkpoint(obs_ckpt)
action_alg, action_state, _, _ = load_checkpoint(action_ckpt)

# Encoding: alg.predict(state, batch, rng) → jax array of shape (B, latent_dim)
z_obs = obs_alg.predict(obs_state, batch, obs_rng)
z_action = action_alg.predict(action_state, batch, action_rng)
```

Scoring function used: `ksg_estimator` from `quality_estimators.py`:
```python
# ks = np.arange(5, 8)  — k values for kNN
score = ksg_estimator(batch, rng, ks, obs_alg, obs_state, action_alg, action_state)
# Returns per-sample score (B,), higher = better quality (more I(obs;action))
```

Checkpoint load format: `load_checkpoint` expects a directory path with:
- `example_batch.msgpack` — example batch for model init
- `config.json` — model config
- Orbax checkpoint subdirs

---

## 5. reference_latents.npz — must be added to train job

The upstream `estimate_quality.py` runs the full scoring loop in one shot; it does NOT save reference latents. Our pipeline requires:
1. `train_job.py` calls `scripts/train.py` (training VAEs)
2. After training, encode the reference set using `DeminfWorker` → save `reference_latents_obs.npy` + `reference_latents_action.npy` to the registry
3. `infer_worker.py` loads these at startup — no re-encoding

This is a new step that must be added.

---

## 6. mcap topic names per embodiment

`# DECISION-NEEDED: real mcap topic names`
- Default (synthetic): `/obs/rgb`, `/action`
- Real names go in `configs/embodiments/<name>.yaml`

---

## 7. DemInf task config (training config)

`# DECISION-NEEDED: DemInf task config`
- Template: `configs/bc/manav.py` in the DemInf repo
- Dataset path must be filled in per task

---

## 8. Robometer model checkpoint

- Default: `robometer/Robometer-4B` (relative to /data/robometer working dir, or HuggingFace Hub ID)
- Fine-tuned checkpoint path: TBD. Configured via `RDF_ROBOMETER_MODEL_PATH` env var.

---

## 9. Training pipeline output format

`keep` episodes are written to clean bucket as:
```
clean/task=<task>/episode=<episode_id>/
  head.mp4
  data.mcap
  metadata.yaml
```

---

## 10. Pipeline mode default

- `sequential` for offline sweeps (all Stage A first, then Stage B)
- `parallel` for live ingestion (Stage B fires when cohort_min reached)
- Default: `sequential`
