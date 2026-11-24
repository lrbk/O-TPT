#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-Flower102}"
SEED="${SEED:-${2:-0}}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/log/trust_region_grid_${DATASET}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-/mlx_devbox/users/zhuziheng.15/playground/miniconda3/envs/otpt/bin/python}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/dataset}"
MODEL_ARCH="${OTPT_MODEL_ARCH:-ViT-B/16}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CTX_INIT="${CTX_INIT:-a_photo_of_a}"

mkdir -p "${RUN_ROOT}/csv" "${RUN_ROOT}/stdout"
printf 'status\tdataset\tseed\tmode\tlambda\tcsv\tstdout\n' > "${RUN_ROOT}/manifest.tsv"

modes=(${TRUST_REGION_MODES:-ctx text both})
lambdas=(${TRUST_REGION_LAMBDAS:-0 0.001 0.003 0.01 0.03 0.1})

for mode in "${modes[@]}"; do
  for lam in "${lambdas[@]}"; do
    tag="mode${mode}_lam${lam//./p}"
    csv="${RUN_ROOT}/csv/trust_region_${DATASET}_seed${SEED}_${tag}.csv"
    stdout="${RUN_ROOT}/stdout/trust_region_${DATASET}_seed${SEED}_${tag}.log"
    echo "START ${DATASET} seed=${SEED} mode=${mode} lambda=${lam}"
    extra_args=()
    if [[ "${lam}" != "0" && "${lam}" != "0.0" ]]; then
      extra_args+=(--use_trust_region --lam_trust_region "${lam}" --trust_region_mode "${mode}")
    fi
    if "${PYTHON_BIN}" "${ROOT_DIR}/otpt_classification_trust_region.py" "${DATA_ROOT}" \
      --test_sets "${DATASET}" \
      --seed "${SEED}" \
      --csv_log "${csv}" \
      -a "${MODEL_ARCH}" \
      -b "${BATCH_SIZE}" \
      --gpu "${GPU_ID}" \
      --tpt \
      --ctx_init "${CTX_INIT}" \
      --run_type tpt_otpt \
      --lambda_term 18 \
      "${extra_args[@]}" > "${stdout}" 2>&1; then
      printf 'OK\t%s\t%s\t%s\t%s\t%s\t%s\n' "${DATASET}" "${SEED}" "${mode}" "${lam}" "${csv}" "${stdout}" >> "${RUN_ROOT}/manifest.tsv"
    else
      printf 'FAIL\t%s\t%s\t%s\t%s\t%s\t%s\n' "${DATASET}" "${SEED}" "${mode}" "${lam}" "${csv}" "${stdout}" >> "${RUN_ROOT}/manifest.tsv"
    fi
  done
done

"${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import csv
import os
import sys

root = sys.argv[1]
rows = []
with open(os.path.join(root, "manifest.tsv")) as fp:
    reader = csv.DictReader(fp, delimiter="\t")
    for item in reader:
        if item["status"] != "OK":
            rows.append({
                "status": item["status"],
                "dataset": item["dataset"],
                "seed": item["seed"],
                "mode": item["mode"],
                "lambda": item["lambda"],
                "acc": "",
                "top5": "",
                "ece": "",
                "csv": item["csv"],
            })
            continue
        vals = {"acc": "", "top5": "", "ece": ""}
        with open(item["csv"], newline="") as f:
            for row in csv.reader(f):
                if row and row[0] == "Accuracy":
                    vals["acc"] = row[1]
                elif row and row[0] == "Top-5 acc.":
                    vals["top5"] = row[1]
                elif row and row[0] == "ECE.":
                    vals["ece"] = row[1]
        rows.append({
            "status": item["status"],
            "dataset": item["dataset"],
            "seed": item["seed"],
            "mode": item["mode"],
            "lambda": item["lambda"],
            "acc": vals["acc"],
            "top5": vals["top5"],
            "ece": vals["ece"],
            "csv": item["csv"],
        })

out = os.path.join(root, "summary.csv")
with open(out, "w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=["status", "dataset", "seed", "mode", "lambda", "acc", "top5", "ece", "csv"])
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY

echo "RUN_ROOT=${RUN_ROOT}"
