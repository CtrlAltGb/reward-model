#!/usr/bin/env bash
# Full pipeline: preprocess → Robometer scoring → DemInf training
#
# Usage:
#   cd /data/reward_model
#   bash scripts/run_with_deminf.sh
#
# Env vars:
#   RDF_N_EPISODES   (default: 30)
#   RDF_INSTRUCTION  (default: "pick the red cube and place in the blue box")
#   RDF_THRESHOLD    (default: 0.5)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DEMINF_DATA="/tmp/rdf_pipeline_deminf/deminf_data"
CKPT_PATH="/tmp/rdf_deminf_ckpts"

# ── Phases 1–5: preprocess + Robometer + symlink ─────────────────────────────
echo ""
echo "Running preprocess_and_score.py ..."
/data/robometer/.venv/bin/python3 -u \
    "$SCRIPT_DIR/preprocess_and_score.py"

# ── Phase 6: DemInf MCAP preprocessing ───────────────────────────────────────
echo ""
echo "============================================================"
echo "Phase 6: DemInf MCAP → npz cache"
echo "============================================================"
cd /data/demonstration-information
/data/.conda/envs/openx/bin/python3 scripts/preprocess_episodes.py \
    --root "$DEMINF_DATA" \
    --splits train test \
    --workers 4

# ── Phase 7: DemInf VAE training ─────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Phase 7: DemInf VAE training"
echo "============================================================"
WANDB_MODE=disabled \
/data/.conda/envs/openx/bin/python3 scripts/train.py \
    --config "$REPO_ROOT/configs/quality/clean_data_vae.py:sa" \
    --path   "$CKPT_PATH" \
    --name   "clean_data_vae"

echo ""
echo "============================================================"
echo "Done. Checkpoint saved under $CKPT_PATH/clean_data_vae/"
echo "============================================================"
