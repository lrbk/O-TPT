#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/dataset}"
TESTSET="${1:-Flower102}"
METHOD="${2:-entropy}"
SEED="${SEED:-${3:-0}}"
MODEL_ARCH="${OTPT_MODEL_ARCH:-ViT-B/16}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CTX_INIT="${CTX_INIT:-a_photo_of_a}"
RUN_TYPE="${RUN_TYPE:-tpt_otpt}"
LAMBDA_TERM="${LAMBDA_TERM:-18}"
PSR_T="${PSR_T:-2.0}"
LAM_IM="${LAM_IM:-0.8}"
CSV_LOG="${CSV_LOG:-${ROOT_DIR}/log/reproduce_${METHOD}_${TESTSET}_seed${SEED}.csv}"
PYTHON_BIN="${PYTHON_BIN:-/mlx_devbox/users/zhuziheng.15/playground/miniconda3/envs/otpt/bin/python}"

mkdir -p "${ROOT_DIR}/log"

cd "${ROOT_DIR}"

exec "${PYTHON_BIN}" ./otpt_classification_psr_im.py "${DATA_ROOT}" \
  --test_sets "${TESTSET}" \
  --seed "${SEED}" \
  --csv_log "${CSV_LOG}" \
  -a "${MODEL_ARCH}" \
  -b "${BATCH_SIZE}" \
  --gpu "${GPU_ID}" \
  --tpt \
  --ctx_init "${CTX_INIT}" \
  --run_type "${RUN_TYPE}" \
  --lambda_term "${LAMBDA_TERM}" \
  --tta_objective "${METHOD}" \
  --psr_T "${PSR_T}" \
  --lam_im "${LAM_IM}"
