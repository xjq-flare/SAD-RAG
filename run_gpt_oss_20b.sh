#!/bin/bash
# =============================================================================
# SAD-RAG Inference Script: GPT-OSS-20B
# =============================================================================
# This script runs the SAD-RAG inference pipeline using GPT-OSS-20B.
# GPT-OSS-20B requires reasoning_effort="low".
# Make sure the vLLM server is running before executing this script.
#
# Usage: bash run_gpt_oss_20b.sh
# =============================================================================

set -e

# ---- Configuration ----
TAU=${TAU:-1}
OPENAI_API_BASE=${OPENAI_API_BASE:-"http://localhost:6006/v1"}
OPENAI_API_KEY=${OPENAI_API_KEY:-"EMPTY"}
MODEL_ID="openai-mirror/gpt-oss-20b"

# GPT-OSS-specific: low reasoning effort
MODEL_REASONING_EFFORT="low"

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
DATA_DIR="${SCRIPT_DIR}/data"
ENTRY_DIR="${SCRIPT_DIR}/argumentative_truth_discovery"

# ---- Dataset files ----
DATASETS=(
    "data_poisoning_dataset_ratio_20.jsonl"
    "data_poisoning_dataset_ratio_40.jsonl"
    "data_poisoning_dataset_ratio_60.jsonl"
    "data_poisoning_dataset_ratio_80.jsonl"
    "data_poisoning_dataset_ratio_90.jsonl"
    "prompt_injection_dataset_ratio_20.jsonl"
    "prompt_injection_dataset_ratio_40.jsonl"
    "prompt_injection_dataset_ratio_60.jsonl"
    "prompt_injection_dataset_ratio_80.jsonl"
    "prompt_injection_dataset_ratio_90.jsonl"
)

# ---- Configure .env ----
echo "============================================="
echo " SAD-RAG Inference: ${MODEL_ID}"
echo "============================================="
echo "Configuring .env ..."
echo "  Note: reasoning_effort = low"

cat > "${ENV_FILE}" << EOF
MODEL_PROVIDER=vllm
MODEL_ID=${MODEL_ID}
OPENAI_API_BASE=${OPENAI_API_BASE}
OPENAI_API_KEY=${OPENAI_API_KEY}
MODEL_EXTRA_BODY=
MODEL_REASONING_EFFORT=${MODEL_REASONING_EFFORT}
EOF

echo "  MODEL_ID              = ${MODEL_ID}"
echo "  OPENAI_API_BASE       = ${OPENAI_API_BASE}"
echo "  TAU                   = ${TAU}"
echo "  MODEL_REASONING_EFFORT = ${MODEL_REASONING_EFFORT}"
echo "============================================="

# ---- Verify datasets exist ----
echo "Checking dataset files ..."
for dataset in "${DATASETS[@]}"; do
    if [ ! -f "${DATA_DIR}/${dataset}" ]; then
        echo "ERROR: Dataset file not found: ${DATA_DIR}/${dataset}"
        exit 1
    fi
done
echo "All dataset files found."
echo ""

# ---- Build dataset path list for __main__.py ----
MAIN_PY="${ENTRY_DIR}/__main__.py"

DATASET_PATHS_PY="["
for dataset in "${DATASETS[@]}"; do
    DATASET_PATHS_PY="${DATASET_PATHS_PY}\"${DATA_DIR}/${dataset}\","
done
DATASET_PATHS_PY="${DATASET_PATHS_PY}]"

# ---- Update __main__.py with dataset paths ----
python3 - << PYEOF
import re

main_py = "${MAIN_PY}"
new_paths = """    dataset_paths = ${DATASET_PATHS_PY}
"""

with open(main_py, 'r') as f:
    content = f.read()

content = re.sub(
    r'    dataset_paths = \[\s*\]',
    new_paths.rstrip(),
    content
)

with open(main_py, 'w') as f:
    f.write(content)

print(f"Updated dataset_paths in {main_py}")
PYEOF

echo "Running inference ..."
echo ""

# ---- Run inference ----
cd "${ENTRY_DIR}"
python3 __main__.py

echo ""
echo "============================================="
echo " Inference completed!"
echo "============================================="