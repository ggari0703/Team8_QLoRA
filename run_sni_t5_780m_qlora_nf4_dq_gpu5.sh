#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

CONFIG="reproduce_table3/configs/sni_local/t5_780m_qlora_nf4_dq.yaml"
GPU_ID="${GPU_ID:-5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d reproduce_table3/data/natural-instructions/tasks ]] || [[ ! -d reproduce_table3/data/natural-instructions/splits/default ]]; then
  echo "Missing local SNI data under reproduce_table3/data/natural-instructions" >&2
  exit 1
fi

mkdir -p reproduce_table3/logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "Running ${CONFIG} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" -m reproduce_table3.experiment_files.runner.run_experiments \
  --config "${CONFIG}" \
  2>&1 | tee "reproduce_table3/logs/sni_t5_780m_qlora_nf4_dq_gpu${GPU_ID}.log"
