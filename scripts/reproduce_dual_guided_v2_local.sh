#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/dataset}"
TESTSET="${1:-pets}"
SEED="${SEED:-${2:-0}}"
MODEL_ARCH="${OTPT_MODEL_ARCH:-ViT-B/16}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CTX_INIT="${CTX_INIT:-a_photo_of_a}"
CSV_LOG="${CSV_LOG:-${ROOT_DIR}/log/dual_guided_v2_${TESTSET}_seed${SEED}.tsv}"
PYTHON_BIN="${PYTHON_BIN:-/mlx_devbox/users/zhuziheng.15/playground/miniconda3/envs/otpt/bin/python}"

mkdir -p "${ROOT_DIR}/log"

cd "${ROOT_DIR}"

exec "${PYTHON_BIN}" ./otpt_classification_dual_guided_v2.py "${DATA_ROOT}" \
  --test_sets "${TESTSET}" \
  --seed "${SEED}" \
  --csv_log "${CSV_LOG}" \
  -a "${MODEL_ARCH}" \
  -b "${BATCH_SIZE}" \
  --gpu "${GPU_ID}" \
  --tpt \
  --ctx_init "${CTX_INIT}" \
  --run_type tpt \
  --eat_pt \
  --view_selection all \
  --gate_init 0.1 \
  --gate_temp 2.0 \
  --eat_visual_loss adaptive_kl \
  --eat_text_loss sample_distill \
  --eat_tau_router 1.0 \
  --eat_visual_conf_threshold 0.55 \
  --eat_visual_mix_temp 0.10 \
  --eat_lambda_disp 0.15 \
  --eat_lambda_anchor 0.05 \
  --eat_anchor_topm 20 \
  --eat_text_topm 20 \
  --eat_lambda_res 0.01
