#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-Flower102}"
METHOD="${2:-a4_nc_full}"
SEED="${3:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${ROOT_DIR}/log/${METHOD}_$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')_seed${SEED}_${TIMESTAMP}.log"
CSV_FILE="${ROOT_DIR}/log/${METHOD}_$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')_seed${SEED}_${TIMESTAMP}.csv"

mkdir -p "${ROOT_DIR}/log"

cd "${ROOT_DIR}"
CSV_LOG="${CSV_FILE}" bash "${ROOT_DIR}/scripts/reproduce_otpt_nc_local.sh" "${DATASET}" "${METHOD}" "${SEED}" > "${LOG_FILE}" 2>&1

printf 'log=%s\ncsv=%s\n' "${LOG_FILE}" "${CSV_FILE}"
