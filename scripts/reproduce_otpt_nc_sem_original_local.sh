#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/dataset}"
TESTSET="${1:-Flower102}"
SEED="${SEED:-${2:-0}}"
METHOD="${3:-nc_sem}"
MODEL_ARCH="${OTPT_MODEL_ARCH:-ViT-B/16}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CTX_INIT="${CTX_INIT:-a_photo_of_a}"
RUN_TYPE="${RUN_TYPE:-tpt_nc_sem}"
LAM_NC_SEM="${LAM_NC_SEM:-1.0}"
LAM_MM_VISUAL="${LAM_MM_VISUAL:-0.2}"
LAM_MM_CROSS="${LAM_MM_CROSS:-0.2}"
SEM_ALPHA="${SEM_ALPHA:-0.5}"
SEM_PRIOR_PATH="${SEM_PRIOR_PATH:-}"
LAM_ENTROPY_TARGET="${LAM_ENTROPY_TARGET:-2.0}"
ENTROPY_TARGET_MARGIN="${ENTROPY_TARGET_MARGIN:-0.0}"
ENTROPY_TARGET_MODE="${ENTROPY_TARGET_MODE:-floor}"
LAM_LOGITNORM="${LAM_LOGITNORM:-0.0}"
LOGITNORM_TAU="${LOGITNORM_TAU:-0.04}"
LAM_TRUST_REGION="${LAM_TRUST_REGION:-0.0}"
TRUST_REGION_MODE="${TRUST_REGION_MODE:-ctx}"
TEMPERATURE_GRID="${TEMPERATURE_GRID:-0}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-}"
PYTHON_BIN="${PYTHON_BIN:-/mlx_devbox/users/zhuziheng.15/playground/miniconda3/envs/otpt/bin/python}"
CSV_LOG="${CSV_LOG:-${ROOT_DIR}/log/reproduce_${METHOD}_original_${TESTSET}_seed${SEED}.csv}"

EXTRA_ARGS=()
case "${METHOD}" in
  nc_sem)
    EXTRA_ARGS+=(--use_nc_sem --lam_nc_sem "${LAM_NC_SEM}" --sem_alpha "${SEM_ALPHA}")
    ;;
  mm_ortho)
    EXTRA_ARGS+=(--use_nc_sem --use_mm_ortho --lam_nc_sem "${LAM_NC_SEM}" --sem_alpha "${SEM_ALPHA}" --lam_mm_visual "${LAM_MM_VISUAL}" --lam_mm_cross "${LAM_MM_CROSS}")
    ;;
  cache_ortho)
    EXTRA_ARGS+=(--use_nc_sem --use_cache_ortho --lam_nc_sem "${LAM_NC_SEM}" --sem_alpha "${SEM_ALPHA}")
    ;;
  sem_ortho)
    EXTRA_ARGS+=(--use_nc_sem --lam_nc_sem "${LAM_NC_SEM}" --sem_alpha 0.0)
    ;;
  dist_align)
    EXTRA_ARGS+=(--use_nc_sem --use_dist_align --lam_nc_sem "${LAM_NC_SEM}" --sem_alpha "${SEM_ALPHA}")
    ;;
  proda_ortho)
    EXTRA_ARGS+=(--use_nc_sem --use_proda_ortho --lam_nc_sem "${LAM_NC_SEM}" --sem_alpha "${SEM_ALPHA}")
    ;;
  adv_ortho)
    EXTRA_ARGS+=(--use_nc_sem --use_adv_ortho --lam_nc_sem "${LAM_NC_SEM}" --sem_alpha "${SEM_ALPHA}")
    ;;
  entropy_target)
    EXTRA_ARGS+=(--use_entropy_target --lam_entropy_target "${LAM_ENTROPY_TARGET}" --entropy_target_margin "${ENTROPY_TARGET_MARGIN}" --entropy_target_mode "${ENTROPY_TARGET_MODE}")
    ;;
  logitnorm_tpt)
    EXTRA_ARGS+=(--use_logitnorm_tpt --lam_logitnorm "${LAM_LOGITNORM}" --logitnorm_tau "${LOGITNORM_TAU}")
    ;;
  trust_region)
    EXTRA_ARGS+=(--use_trust_region --lam_trust_region "${LAM_TRUST_REGION}" --trust_region_mode "${TRUST_REGION_MODE}")
    ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 1
    ;;
esac
if [[ -n "${SEM_PRIOR_PATH}" ]]; then
  EXTRA_ARGS+=(--sem_prior_path "${SEM_PRIOR_PATH}")
fi
if [[ -n "${MAX_TEST_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--max_test_samples "${MAX_TEST_SAMPLES}")
fi
if [[ "${TEMPERATURE_GRID}" == "1" ]]; then
  EXTRA_ARGS+=(--temperature_grid)
fi

mkdir -p "${ROOT_DIR}/log"

cd "${ROOT_DIR}"

exec "${PYTHON_BIN}" ./otpt_classification_nc_sem.py "${DATA_ROOT}" \
  --test_sets "${TESTSET}" \
  --seed "${SEED}" \
  --csv_log "${CSV_LOG}" \
  -a "${MODEL_ARCH}" \
  -b "${BATCH_SIZE}" \
  --gpu "${GPU_ID}" \
  --tpt \
  --ctx_init "${CTX_INIT}" \
  --run_type "${RUN_TYPE}" \
  "${EXTRA_ARGS[@]}"
