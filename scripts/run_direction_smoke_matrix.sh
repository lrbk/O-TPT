#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${1:-${ROOT_DIR}/log/direction_smoke_$(date +%Y%m%d_%H%M%S)}"
WAIT_PID="${2:-}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-200}"

mkdir -p "${RUN_ROOT}/stdout" "${RUN_ROOT}/csv"
printf 'status\tmethod\tdataset\tseed\tcsv\tstdout\n' > "${RUN_ROOT}/manifest.tsv"

if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
fi

methods=(
  nc_sem
  mm_ortho
  cache_ortho
  sem_ortho
  dist_align
  proda_ortho
  adv_ortho
)

datasets=(Flower102 Pets)

for method in "${methods[@]}"; do
  for dataset in "${datasets[@]}"; do
    csv="${RUN_ROOT}/csv/${method}_${dataset}_seed0_subset${MAX_TEST_SAMPLES}.csv"
    stdout="${RUN_ROOT}/stdout/${method}_${dataset}_seed0_subset${MAX_TEST_SAMPLES}.log"
    if MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES}" CSV_LOG="${csv}" \
      bash "${ROOT_DIR}/scripts/reproduce_otpt_nc_sem_original_local.sh" "${dataset}" 0 "${method}" > "${stdout}" 2>&1; then
      printf 'OK\t%s\t%s\t0\t%s\t%s\n' "${method}" "${dataset}" "${csv}" "${stdout}" >> "${RUN_ROOT}/manifest.tsv"
    else
      printf 'FAIL\t%s\t%s\t0\t%s\t%s\n' "${method}" "${dataset}" "${csv}" "${stdout}" >> "${RUN_ROOT}/manifest.tsv"
    fi
  done
done
