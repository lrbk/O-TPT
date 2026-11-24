#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS="${DATASETS:-Flower102 Pets}"
METHODS="${METHODS:-entropy psr im psr_im}"
SEED="${SEED:-0}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/log/psr_im_${RUN_TS}}"
MANIFEST="${RUN_ROOT}/manifest.tsv"

mkdir -p "${RUN_ROOT}/stdout" "${RUN_ROOT}/csv"
printf 'status\tdataset\tmethod\tseed\tcsv\tstdout\n' > "${MANIFEST}"

for dataset in ${DATASETS}; do
  for method in ${METHODS}; do
    dataset_slug="$(echo "${dataset}" | tr '[:upper:]' '[:lower:]')"
    csv_log="${RUN_ROOT}/csv/${method}_${dataset_slug}_seed${SEED}.csv"
    stdout_log="${RUN_ROOT}/stdout/${method}_${dataset_slug}_seed${SEED}.log"
    echo "===== dataset=${dataset} method=${method} seed=${SEED} ====="
    if bash "${ROOT_DIR}/scripts/reproduce_otpt_psr_im_local.sh" "${dataset}" "${method}" "${SEED}" > "${stdout_log}" 2>&1; then
      printf 'OK\t%s\t%s\t%s\t%s\t%s\n' "${dataset}" "${method}" "${SEED}" "${csv_log}" "${stdout_log}" >> "${MANIFEST}"
    else
      printf 'FAIL\t%s\t%s\t%s\t%s\t%s\n' "${dataset}" "${method}" "${SEED}" "${csv_log}" "${stdout_log}" >> "${MANIFEST}"
      exit 1
    fi
  done
done

echo "run_root=${RUN_ROOT}"
