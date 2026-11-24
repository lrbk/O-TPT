#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTSET="${1:-Flower102}"
SEED="${SEED:-${2:-0}}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/log/worker_adaptive_lambda_${TESTSET}_seed${SEED}_${TIMESTAMP}.log}"
PID_FILE="${PID_FILE:-${ROOT_DIR}/log/worker_adaptive_lambda_${TESTSET}_seed${SEED}.pid}"
LATEST_LOG_FILE="${ROOT_DIR}/log/worker_adaptive_lambda_${TESTSET}_seed${SEED}.latest"

mkdir -p "${ROOT_DIR}/log"

nohup bash "${ROOT_DIR}/scripts/reproduce_otpt_adaptive_lambda_local.sh" "${TESTSET}" "${SEED}" > "${LOG_FILE}" 2>&1 &
echo $! | tee "${PID_FILE}"
printf '%s\n' "${LOG_FILE}" > "${LATEST_LOG_FILE}"
printf 'log=%s\npid=%s\n' "${LOG_FILE}" "$(cat "${PID_FILE}")"
