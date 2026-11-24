#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=/mlx_devbox/users/zhuziheng.15/playground/O-TPT
RUN_ROOT="$1"
printf 'status\tdataset\tseed\tcsv\tstdout\n' > "$RUN_ROOT/manifest.tsv"
for dataset in Flower102 Pets; do
  csv="$RUN_ROOT/csv/nc_sem_original_${dataset}_seed0.csv"
  stdout="$RUN_ROOT/stdout/nc_sem_original_${dataset}_seed0.log"
  if CSV_LOG="$csv" LAM_NC_SEM=1.0 SEM_ALPHA=0.5 bash "$ROOT_DIR/scripts/reproduce_otpt_nc_sem_original_local.sh" "$dataset" 0 > "$stdout" 2>&1; then
    printf 'OK\t%s\t0\t%s\t%s\n' "$dataset" "$csv" "$stdout" >> "$RUN_ROOT/manifest.tsv"
  else
    printf 'FAIL\t%s\t0\t%s\t%s\n' "$dataset" "$csv" "$stdout" >> "$RUN_ROOT/manifest.tsv"
    exit 1
  fi
done
