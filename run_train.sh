#!/bin/bash
set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-/data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/python}
echo ">>> Starting exp_infofusion_2026 training (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
"$PYTHON_BIN" training/train_bio_cot_v3.2.py "$@"
