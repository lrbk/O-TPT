#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTSET="${1:-pets}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/log/worker_reproduce_${TESTSET}_${TIMESTAMP}.log}"
PID_FILE="${PID_FILE:-${ROOT_DIR}/log/worker_reproduce_${TESTSET}.pid}"
LATEST_LOG_FILE="${ROOT_DIR}/log/worker_reproduce_${TESTSET}.latest"

mkdir -p "${ROOT_DIR}/log"

nohup bash "${ROOT_DIR}/scripts/reproduce_otpt_local.sh" "${TESTSET}" > "${LOG_FILE}" 2>&1 &
echo $! | tee "${PID_FILE}"
printf '%s\n' "${LOG_FILE}" > "${LATEST_LOG_FILE}"
printf 'log=%s\npid=%s\n' "${LOG_FILE}" "$(cat "${PID_FILE}")"
