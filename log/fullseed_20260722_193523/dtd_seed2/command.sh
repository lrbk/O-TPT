#!/usr/bin/env bash
set -euo pipefail
cd "/mlx_devbox/users/zhuziheng.15/playground/O-TPT"
exec env   CSV_LOG="/mlx_devbox/users/zhuziheng.15/playground/O-TPT/log/fullseed_20260722_193523/dtd_seed2/result.csv"   SEED="2"   GPU_ID="0"   OTPT_MODEL_ARCH="ViT-B/16"   RUN_TYPE="tpt_otpt"   LAMBDA_TERM="18"   BATCH_SIZE="64"   CTX_INIT="a_photo_of_a"   PYTHON_BIN="/mlx_devbox/users/zhuziheng.15/playground/miniconda3/envs/otpt/bin/python"   bash "/mlx_devbox/users/zhuziheng.15/playground/O-TPT/scripts/reproduce_otpt_local.sh" "DTD"
