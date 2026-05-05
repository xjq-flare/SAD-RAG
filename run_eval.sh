#!/bin/bash
# =============================================================================
# SAD-RAG Evaluation Script
# =============================================================================
# This script runs the ASR (Attack Success Rate) and ACC (Answer Accuracy)
# evaluation using qwen/qwen3-235b-a22b-2507 as the judge model.
#
# You must first configure the evaluation .env file with your API credentials.
#
# Usage: bash run_eval.sh [result_dir]
#   result_dir: path to the directory containing inference result JSONL files
#               (default: the most recent record directory)
# =============================================================================

set -e

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="${SCRIPT_DIR}/argumentative_truth_discovery/evaluation"
ENV_FILE="${EVAL_DIR}/.env"
DATA_DIR="${SCRIPT_DIR}/data"

# ---- Check evaluation .env ----
if [ ! -f "${ENV_FILE}" ]; then
    echo "ERROR: Evaluation .env not found at ${ENV_FILE}"
    echo "Please create it with your API credentials:"
    echo "  MODEL_ID=qwen/qwen3-235b-a22b-2507"
    echo "  OPENAI_API_BASE=your-api-base-url"
    echo "  OPENAI_API_KEY=your-api-key"
    exit 1
fi

# ---- Find result directory ----
if [ -n "$1" ]; then
    RESULT_DIR="$1"
else
    # Find the most recent result directory in argumentative_truth_discovery/record/
    RECORD_DIR="${SCRIPT_DIR}/argumentative_truth_discovery/record"
    if [ ! -d "${RECORD_DIR}" ]; then
        echo "ERROR: No record directory found at ${RECORD_DIR}"
        echo "Please run inference first, or specify the result directory:"
        echo "  bash run_eval.sh /path/to/result_dir"
        exit 1
    fi
    RESULT_DIR=$(ls -dt "${RECORD_DIR}"/*/ 2>/dev/null | head -1)
    if [ -z "${RESULT_DIR}" ]; then
        echo "ERROR: No result directories found in ${RECORD_DIR}"
        echo "Please run inference first."
        exit 1
    fi
    RESULT_DIR="${RESULT_DIR%/}"  # Remove trailing slash
fi

echo "============================================="
echo " SAD-RAG Evaluation"
echo "============================================="
echo "  Result directory: ${RESULT_DIR}"
echo "  Data directory:   ${DATA_DIR}"
echo ""

# ---- Collect result files ----
RESULT_FILES=()
for dataset in \
    data_poisoning_dataset_ratio_20_results.jsonl \
    data_poisoning_dataset_ratio_40_results.jsonl \
    data_poisoning_dataset_ratio_60_results.jsonl \
    data_poisoning_dataset_ratio_80_results.jsonl \
    data_poisoning_dataset_ratio_90_results.jsonl \
    prompt_injection_dataset_ratio_20_results.jsonl \
    prompt_injection_dataset_ratio_40_results.jsonl \
    prompt_injection_dataset_ratio_60_results.jsonl \
    prompt_injection_dataset_ratio_80_results.jsonl \
    prompt_injection_dataset_ratio_90_results.jsonl; do
    
    result_file="${RESULT_DIR}/${dataset}"
    if [ -f "${result_file}" ]; then
        RESULT_FILES+=("\"${result_file}\"")
    else
        echo "  WARNING: Result file not found: ${result_file}"
    fi
done

if [ ${#RESULT_FILES[@]} -eq 0 ]; then
    echo "ERROR: No result files found in ${RESULT_DIR}"
    exit 1
fi

echo "Found ${#RESULT_FILES[@]} result files to evaluate."
echo ""

# ---- Update evaluation script with file paths ----
EVAL_SCRIPT="${EVAL_DIR}/asr_and_acc_batch.py"

# Generate the test_file_paths and dataset_path_pattern
python3 - << PYEOF
import re

eval_script = "${EVAL_SCRIPT}"
data_dir = "${DATA_DIR}"
result_dir = "${RESULT_DIR}"

# Build test_file_paths list
result_files = []
datasets = [
    "data_poisoning_dataset_ratio_20_results.jsonl",
    "data_poisoning_dataset_ratio_40_results.jsonl",
    "data_poisoning_dataset_ratio_60_results.jsonl",
    "data_poisoning_dataset_ratio_80_results.jsonl",
    "data_poisoning_dataset_ratio_90_results.jsonl",
    "prompt_injection_dataset_ratio_20_results.jsonl",
    "prompt_injection_dataset_ratio_40_results.jsonl",
    "prompt_injection_dataset_ratio_60_results.jsonl",
    "prompt_injection_dataset_ratio_80_results.jsonl",
    "prompt_injection_dataset_ratio_90_results.jsonl",
]
for d in datasets:
    path = f"{result_dir}/{d}"
    result_files.append(f'        "{path}",')

test_file_paths = "    test_file_paths = [\n" + "\n".join(result_files) + "\n    ]"
dataset_path_pattern = f'    dataset_path_pattern = "{data_dir}/*.jsonl"'

with open(eval_script, 'r') as f:
    content = f.read()

# Replace test_file_paths
content = re.sub(
    r'    test_file_paths = \[.*?\]',
    test_file_paths,
    content,
    flags=re.DOTALL
)

# Replace dataset_path_pattern
content = re.sub(
    r'    dataset_path_pattern = ""',
    dataset_path_pattern,
    content
)

with open(eval_script, 'w') as f:
    f.write(content)

print(f"Updated {eval_script}")
PYEOF

# ---- Run evaluation ----
echo "Running evaluation ..."
echo ""

cd "${EVAL_DIR}"
python3 asr_and_acc_batch.py

echo ""
echo "============================================="
echo " Evaluation completed!"
echo "============================================="