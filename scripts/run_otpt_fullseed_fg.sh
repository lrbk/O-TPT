#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/log/fullseed_${RUN_TS}}"
MANIFEST="${RUN_ROOT}/manifest.tsv"
LATEST_FILE="${ROOT_DIR}/log/latest_fullseed_run_dir.txt"

DATASETS="${DATASETS:-pets Flower102 DTD}"
SEEDS="${SEEDS:-0 1 2}"
GPU_ID="${GPU_ID:-0}"
MODEL_ARCH="${OTPT_MODEL_ARCH:-ViT-B/16}"
RUN_TYPE="${RUN_TYPE:-tpt_otpt}"
LAMBDA_TERM="${LAMBDA_TERM:-18}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CTX_INIT="${CTX_INIT:-a_photo_of_a}"
PYTHON_BIN="${PYTHON_BIN:-/mlx_devbox/users/zhuziheng.15/playground/miniconda3/envs/otpt/bin/python}"

mkdir -p "${RUN_ROOT}"
printf '%s\n' "${RUN_ROOT}" > "${LATEST_FILE}"
printf 'dataset\tseed\trun_dir\tstatus\tresult_csv\tstdout\tstderr\n' > "${MANIFEST}"

run_one() {
  local dataset="$1"
  local seed="$2"
  local dataset_slug
  dataset_slug="$(echo "${dataset}" | tr '[:upper:]' '[:lower:]')"
  local run_dir="${RUN_ROOT}/${dataset_slug}_seed${seed}"
  local stdout_log="${run_dir}/stdout.log"
  local stderr_log="${run_dir}/stderr.log"
  local result_csv="${run_dir}/result.csv"
  local status_file="${run_dir}/status.txt"
  local exit_code_file="${run_dir}/exit_code.txt"
  local command_file="${run_dir}/command.sh"

  mkdir -p "${run_dir}"

  cat > "${command_file}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT_DIR}"
exec env \
  CSV_LOG="${result_csv}" \
  SEED="${seed}" \
  GPU_ID="${GPU_ID}" \
  OTPT_MODEL_ARCH="${MODEL_ARCH}" \
  RUN_TYPE="${RUN_TYPE}" \
  LAMBDA_TERM="${LAMBDA_TERM}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  CTX_INIT="${CTX_INIT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${ROOT_DIR}/scripts/reproduce_otpt_local.sh" "${dataset}"
EOF
  chmod +x "${command_file}"

  echo running > "${status_file}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${dataset}" "${seed}" "${run_dir}" "running" "${result_csv}" "${stdout_log}" "${stderr_log}" >> "${MANIFEST}"

  set +e
  bash "${command_file}" >"${stdout_log}" 2>"${stderr_log}"
  local rc=$?
  set -e

  printf '%s\n' "${rc}" > "${exit_code_file}"
  if [ "${rc}" -eq 0 ]; then
    echo success > "${status_file}"
  else
    echo failed > "${status_file}"
  fi
}

for dataset in ${DATASETS}; do
  for seed in ${SEEDS}; do
    echo "===== dataset=${dataset} seed=${seed} ====="
    run_one "${dataset}" "${seed}"
  done
done

echo "all_done"
