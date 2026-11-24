#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTSET="${1:-pets}"
LOG_FILE="${ROOT_DIR}/log/worker_reproduce_${TESTSET}.log"
PID_FILE="${ROOT_DIR}/log/worker_reproduce_${TESTSET}.pid"

echo "[cwd] ${ROOT_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  echo "[pid]"
  cat "${PID_FILE}"
fi

echo "[process]"
ps -ef | grep otpt_classification | grep -v grep || true

echo "[log_tail]"
tail -n 40 "${LOG_FILE}" || true

echo "[gpu]"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true
