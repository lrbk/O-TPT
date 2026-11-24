#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON=${PYTHON:-/mlx_devbox/users/zhuziheng.15/playground/miniconda3/envs/otpt/bin/python}
DATA_ROOT=${DATA_ROOT:-/mlx_devbox/users/zhuziheng.15/playground/dataset}
DATASETS=${DATASETS:-"Flower102 Pets"}
SEEDS=${SEEDS:-"0 1 2"}
METHODS=${METHODS:-"e0_otpt e2_visual_adaptive_kl e3_text_sample_distill e4_visual_text_dual"}
GPU_ID=${GPU_ID:-0}
CLIP_ARCH=${CLIP_ARCH:-ViT-B/16}
BATCH_SIZE=${BATCH_SIZE:-64}
CTX_INIT=${CTX_INIT:-a_photo_of_a}
TTA_STEPS=${TTA_STEPS:-1}
SELECTION_P=${SELECTION_P:-0.1}
LR=${LR:-5e-3}
PRINT_FREQ=${PRINT_FREQ:-200}
WORKERS=${WORKERS:-4}
DATASET_MODE=${DATASET_MODE:-test}
LOG_ROOT=${LOG_ROOT:-./log/dual_guided_v2_$(date +%Y%m%d_%H%M%S)}
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
    "${PYTHON}" ./otpt_classification_dual_guided_v2.py "${DATA_ROOT}"
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

run_e0_otpt() {
  run_cmd "e0_otpt" "$1" "$2" \
    --tpt \
    --run_type tpt_otpt \
    --view_selection hard_toprho \
    --lambda_term 18
}

run_e2_visual_adaptive_kl() {
  run_cmd "e2_visual_adaptive_kl" "$1" "$2" \
    --tpt \
    --run_type tpt \
    --eat_pt \
    --view_selection hard_toprho \
    --gate_init 0.1 \
    --gate_temp 2.0 \
    --eat_lambda_res 0.01 \
    --eat_visual_loss adaptive_kl \
    --eat_text_loss none \
    --eat_tau_router 1.0 \
    --eat_lambda_disp 0.0
}

run_e3_text_sample_distill() {
  run_cmd "e3_text_sample_distill" "$1" "$2" \
    --tpt \
    --run_type tpt \
    --eat_pt \
    --view_selection hard_toprho \
    --gate_init 0.1 \
    --gate_temp 2.0 \
    --eat_visual_loss tpt_entropy \
    --eat_text_loss sample_distill \
    --eat_tau_router 1.0 \
    --eat_lambda_disp 0.0 \
    --eat_lambda_anchor 0.0 \
    --eat_anchor_topm 20 \
    --eat_text_topm 20 \
    --eat_lambda_res 0.01
}

run_e4_visual_text_dual() {
  run_cmd "e4_visual_text_dual" "$1" "$2" \
    --tpt \
    --run_type tpt \
    --eat_pt \
    --view_selection hard_toprho \
    --gate_init 0.1 \
    --gate_temp 2.0 \
    --eat_visual_loss adaptive_kl \
    --eat_text_loss sample_distill \
    --eat_tau_router 1.0 \
    --eat_lambda_disp 0.0 \
    --eat_lambda_anchor 0.0 \
    --eat_anchor_topm 20 \
    --eat_text_topm 20 \
    --eat_lambda_res 0.01
}

for dataset in ${DATASETS}; do
  for seed in ${SEEDS}; do
    for method in ${METHODS}; do
      case "${method}" in
        e0_otpt)
          run_e0_otpt "${dataset}" "${seed}"
          ;;
        e2_visual_adaptive_kl)
          run_e2_visual_adaptive_kl "${dataset}" "${seed}"
          ;;
        e3_text_sample_distill)
          run_e3_text_sample_distill "${dataset}" "${seed}"
          ;;
        e4_visual_text_dual)
          run_e4_visual_text_dual "${dataset}" "${seed}"
          ;;
        *)
          echo "Unknown method: ${method}" >&2
          exit 1
          ;;
      esac
    done
  done
done

echo "Dual-Guided 2.0 experiments finished."
echo "Logs: ${LOG_ROOT}"
echo "Manifest: ${MANIFEST}"
