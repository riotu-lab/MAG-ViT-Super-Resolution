#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "Usage: $0 MODEL CHECKPOINT LR_X4_DIR MAGVIT_X4_DIR FUNSR_X4_DIR OUTPUT_DIR [DEVICE]" >&2
  echo "MODEL must be maxvit_t or resnext101_32x8d." >&2
  exit 2
fi

classifier_model="$1"
classifier_checkpoint="$2"
lr_x4_dir="$3"
magvit_x4_dir="$4"
funsr_x4_dir="$5"
results_dir="$6"
compute_device="${7:-cuda:0}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${script_dir}/evaluate_aid_x4.py" \
  --model "${classifier_model}" \
  --checkpoint "${classifier_checkpoint}" \
  --lr-x4 "${lr_x4_dir}" \
  --magvit-x4 "${magvit_x4_dir}" \
  --funsr-x4 "${funsr_x4_dir}" \
  --output-dir "${results_dir}" \
  --device "${compute_device}" \
  --expected-images 2000
