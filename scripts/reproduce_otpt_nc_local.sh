#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/dataset}"
TESTSET="${1:-Flower102}"
METHOD="${2:-nc_full}"
SEED="${SEED:-${3:-0}}"
MODEL_ARCH="${OTPT_MODEL_ARCH:-ViT-B/16}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CTX_INIT="${CTX_INIT:-a_photo_of_a}"
PSR_T="${PSR_T:-2.0}"
LAM_IM="${LAM_IM:-0.8}"
LAMBDA_TERM="${LAMBDA_TERM:-18}"
LAM_ETF="${LAM_ETF:-1.0}"
LAM_ALIGN="${LAM_ALIGN:-0.1}"
LAM_ANCHOR="${LAM_ANCHOR:-0.05}"
LAM_PULL="${LAM_PULL:-0.01}"
PYTHON_BIN="${PYTHON_BIN:-/mlx_devbox/users/zhuziheng.15/playground/miniconda3/envs/otpt/bin/python}"
CSV_LOG="${CSV_LOG:-${ROOT_DIR}/log/reproduce_${METHOD}_${TESTSET}_seed${SEED}.csv}"

RUN_TYPE="tpt"
EXTRA_ARGS=()

case "${METHOD}" in
  a1_tpt|entropy)
    RUN_TYPE="tpt"
    ;;
  a2_otpt|otpt)
    RUN_TYPE="tpt_otpt"
    ;;
  a3_nc_etf|nc_etf)
    RUN_TYPE="tpt"
    EXTRA_ARGS+=(--use_etf --lam_etf "${LAM_ETF}" --lam_align 0.0)
    ;;
  a4_nc_full|nc_full)
    RUN_TYPE="tpt"
    EXTRA_ARGS+=(--use_etf --lam_etf "${LAM_ETF}" --lam_align "${LAM_ALIGN}")
    ;;
  a5_anchor_prior|anchor_prior)
    RUN_TYPE="tpt"
    EXTRA_ARGS+=(--use_anchor_prior --lam_anchor "${LAM_ANCHOR}" --lam_pull "${LAM_PULL}" --anchor_topm 20 --anchor_conf_gate)
    ;;
  *)
    echo "Unknown NC method: ${METHOD}" >&2
    exit 1
    ;;
esac

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
  --tta_objective entropy \
  --psr_T "${PSR_T}" \
  --lam_im "${LAM_IM}" \
  "${EXTRA_ARGS[@]}"
