#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/log/psr_im_single_seed_${TIMESTAMP}.log}"
PID_FILE="${PID_FILE:-${ROOT_DIR}/log/psr_im_single_seed.pid}"
LATEST_LOG_FILE="${ROOT_DIR}/log/psr_im_single_seed.latest"

mkdir -p "${ROOT_DIR}/log"

nohup bash "${ROOT_DIR}/scripts/run_psr_im_single_seed.sh" > "${LOG_FILE}" 2>&1 &
echo $! | tee "${PID_FILE}"
printf '%s\n' "${LOG_FILE}" > "${LATEST_LOG_FILE}"
printf 'log=%s\npid=%s\n' "${LOG_FILE}" "$(cat "${PID_FILE}")"
