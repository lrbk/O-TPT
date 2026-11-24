#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-/mlx_devbox/users/zhuziheng.15/playground/dataset}
DATASETS=${DATASETS:-"Flower102 Pets"}
SEEDS=${SEEDS:-"0 1 2"}
METHODS=${METHODS:-"uniform_allviews kl_allviews safe_kl_v3 safe_kl_v3_textres"}
GPU_ID=${GPU_ID:-0}
CLIP_ARCH=${CLIP_ARCH:-ViT-B/16}
BATCH_SIZE=${BATCH_SIZE:-8}
CTX_INIT=${CTX_INIT:-a_photo_of_a}
TTA_STEPS=${TTA_STEPS:-1}
SELECTION_P=${SELECTION_P:-0.1}
LR=${LR:-5e-3}
PRINT_FREQ=${PRINT_FREQ:-200}
WORKERS=${WORKERS:-4}
DATASET_MODE=${DATASET_MODE:-test}
LOG_ROOT=${LOG_ROOT:-./log/safe_anchor_v3_$(date +%Y%m%d_%H%M%S)}
DRY_RUN=${DRY_RUN:-0}
RESUME=${RESUME:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "${LOG_ROOT}/stdout" "${LOG_ROOT}/csv"
MANIFEST="${LOG_ROOT}/manifest.tsv"

if [[ ! -f "${MANIFEST}" ]]; then
  printf "status\texperiment\tdataset\tseed\tcsv\tlog\n" > "${MANIFEST}"
fi

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr '/: ,' '____'
}

run_cmd() {
  local experiment=$1
  local dataset=$2
  local seed=$3
  shift 3

  local dataset_slug csv stdout_log
  dataset_slug="$(slugify "${dataset}")"
  csv="${LOG_ROOT}/csv/${experiment}_${dataset_slug}_seed${seed}.tsv"
  stdout_log="${LOG_ROOT}/stdout/${experiment}_${dataset_slug}_seed${seed}.log"

  if [[ "${RESUME}" == "1" && -s "${csv}" ]]; then
    echo "[SKIP] ${experiment} dataset=${dataset} seed=${seed}"
    return 0
  fi

  local cmd=(
    "${PYTHON}" ./otpt_classification.py "${DATA_ROOT}"
    --test_sets "${dataset}"
    --csv_log "${csv}"
    --dataset_mode "${DATASET_MODE}"
    -a "${CLIP_ARCH}"
    -b "${BATCH_SIZE}"
    --lr "${LR}"
    --gpu "${GPU_ID}"
    --workers "${WORKERS}"
    --ctx_init "${CTX_INIT}"
    --tta_steps "${TTA_STEPS}"
    --selection_p "${SELECTION_P}"
    --print-freq "${PRINT_FREQ}"
    --seed "${seed}"
    "$@"
  )

  echo "===== ${experiment} | dataset=${dataset} | seed=${seed} ====="
  echo "${cmd[*]} ${EXTRA_ARGS}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "DRY_RUN\t%s\t%s\t%s\t%s\t%s\n" "${experiment}" "${dataset}" "${seed}" "${csv}" "${stdout_log}" >> "${MANIFEST}"
    return 0
  fi

  if "${cmd[@]}" ${EXTRA_ARGS} 2>&1 | tee "${stdout_log}"; then
    printf "OK\t%s\t%s\t%s\t%s\t%s\n" "${experiment}" "${dataset}" "${seed}" "${csv}" "${stdout_log}" >> "${MANIFEST}"
  else
    printf "FAIL\t%s\t%s\t%s\t%s\t%s\n" "${experiment}" "${dataset}" "${seed}" "${csv}" "${stdout_log}" >> "${MANIFEST}"
    return 1
  fi
}

run_visual_only() {
  local experiment=$1
  local dataset=$2
  local seed=$3
  local visual_loss=$4
  shift 4
  run_cmd "${experiment}" "${dataset}" "${seed}" \
    --tpt \
    --run_type tpt \
    --eat_pt \
    --view_selection all \
    --gate_init 0.1 \
    --gate_temp 2.0 \
    --eat_lambda_res 0.01 \
    --eat_visual_loss "${visual_loss}" \
    --eat_text_loss none \
    --eat_tau_router 1.0 \
    --eat_lambda_disp 0.0 \
    "$@"
}

run_safe_textres() {
  local experiment=$1
  local dataset=$2
  local seed=$3
  run_cmd "${experiment}" "${dataset}" "${seed}" \
    --tpt \
    --run_type tpt \
    --eat_pt \
    --view_selection all \
    --gate_init 0.1 \
    --gate_temp 2.0 \
    --eat_lambda_res 0.01 \
    --eat_visual_loss safe_kl \
    --eat_text_loss residual_anchor \
    --eat_lambda_disp 0.05 \
    --eat_tau_router 1.0 \
    --eat_lambda_anchor 0.02 \
    --eat_text_topm 20 \
    --eat_safe_gain_margin 0.0 \
    --eat_safe_gain_temp 0.02 \
    --eat_safe_min_ess 0.35 \
    --eat_safe_ess_temp 0.08
}

for dataset in ${DATASETS}; do
  for seed in ${SEEDS}; do
    for method in ${METHODS}; do
      case "${method}" in
        uniform_allviews)
          run_visual_only "visual_uniform_allviews" "${dataset}" "${seed}" "uniform"
          ;;
        kl_allviews)
          run_visual_only "visual_kl_soft_allviews" "${dataset}" "${seed}" "kl_soft"
          ;;
        safe_kl_v3)
          run_visual_only "safe_kl_v3" "${dataset}" "${seed}" "safe_kl" \
            --eat_lambda_anchor 0.02 \
            --eat_safe_gain_margin 0.0 \
            --eat_safe_gain_temp 0.02 \
            --eat_safe_min_ess 0.35 \
            --eat_safe_ess_temp 0.08
          ;;
        safe_kl_v3_textres)
          run_safe_textres "safe_kl_v3_textres" "${dataset}" "${seed}"
          ;;
        *)
          echo "Unknown method: ${method}" >&2
          exit 1
          ;;
      esac
    done
  done
done

echo "Safe anchor v3 experiments finished."
echo "Logs: ${LOG_ROOT}"
echo "Manifest: ${MANIFEST}"
