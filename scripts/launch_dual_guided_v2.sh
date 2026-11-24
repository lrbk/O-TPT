#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/log/dual_guided_v2_launcher_${TIMESTAMP}.log}"
PID_FILE="${PID_FILE:-${ROOT_DIR}/log/dual_guided_v2_batch.pid}"
LATEST_LOG_FILE="${ROOT_DIR}/log/dual_guided_v2_launcher.latest"

mkdir -p "${ROOT_DIR}/log"

nohup bash "${ROOT_DIR}/scripts/run_dual_guided_v2.sh" > "${LOG_FILE}" 2>&1 &
echo $! | tee "${PID_FILE}"
printf '%s\n' "${LOG_FILE}" > "${LATEST_LOG_FILE}"
printf 'log=%s\npid=%s\n' "${LOG_FILE}" "$(cat "${PID_FILE}")"
