#!/usr/bin/env bash
# Usage:
#   ./main.sh
#   DOMAIN=medmcqa ./main.sh
#   DOMAIN=medmcqa RUN_TRAIN=0 ./main.sh   # test only (needs existing checkpoint)
set -euo pipefail

# ============ Adjust targets here ============
# Domain must match a config file: configs/<DOMAIN>.yaml
# Available: amazon | medmcqa | bigbench | safety | grounding
DOMAIN="${DOMAIN:-amazon}"

METHOD="${METHOD:-metaspo}"

# Base model for evaluation (inner/outer loop scoring)
MODEL_TYPE="${MODEL_TYPE:-vllm}"   # openai | vllm
MODEL_NAME="${MODEL_NAME:-Qwen3_8B}"
# MODEL_NAME examples: gpt-4o-mini | llama3.1_8B | llama3.2_3B | Qwen2.5_7B | Qwen3_8B

# Optimizer model (OpenAI API for prompt rewriting) — set in .env or override:
# OPTIM_MODEL_TYPE="${OPTIM_MODEL_TYPE:-openai}"
# OPTIM_MODEL_NAME="${OPTIM_MODEL_NAME:-gpt-4o-mini}"

INIT_SYSTEM_PROMPT="${INIT_SYSTEM_PROMPT:-./prompts/default.json}"
DATASET_DIR="${DATASET_DIR:-./datasets}"
TASK_CONFIG="${TASK_CONFIG:-./configs/${DOMAIN}.yaml}"

# Which stages to run (1=yes, 0=no)
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST_UNSEEN="${RUN_TEST_UNSEEN:-1}"
RUN_TEST_TTA="${RUN_TEST_TTA:-1}"
# ============================================

LOG_ROOT="./logs/${METHOD}/${DOMAIN}"
# save_data() writes one level up: ./logs/<METHOD>/bilevel_nodes_*.json
CHECKPOINT="./logs/${METHOD}/bilevel_nodes_0.json"

if [[ ! -f "${TASK_CONFIG}" ]]; then
  echo "Error: task config not found: ${TASK_CONFIG}" >&2
  echo "Set DOMAIN to one of: amazon medmcqa bigbench safety grounding" >&2
  exit 1
fi

COMMON_ARGS=(
  --task_config_path "${TASK_CONFIG}"
  --dataset_dir "${DATASET_DIR}"
  --base_model_type "${MODEL_TYPE}"
  --base_model_name "${MODEL_NAME}"
)

# Uncomment to override optimizer (defaults: openai + gpt-4o-mini in meta_train.py)
# COMMON_ARGS+=(
#   --optim_model_type "${OPTIM_MODEL_TYPE}"
#   --optim_model_name "${OPTIM_MODEL_NAME}"
# )

echo "DOMAIN=${DOMAIN}  TASK_CONFIG=${TASK_CONFIG}"
echo "MODEL=${MODEL_TYPE}/${MODEL_NAME}  METHOD=${METHOD}"
echo "LOG_ROOT=${LOG_ROOT}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo ">>> Meta-train"
  python meta_train.py \
    --method "${METHOD}" \
    --init_system_prompt_path "${INIT_SYSTEM_PROMPT}" \
    --log_dir "${LOG_ROOT}" \
    "${COMMON_ARGS[@]}"
  # Saves optimized system prompt to ${CHECKPOINT}
else
  echo ">>> Skip meta-train (RUN_TRAIN=0)"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Error: checkpoint not found: ${CHECKPOINT}" >&2
  echo "Run with RUN_TRAIN=1 first, or set INIT_SYSTEM_PROMPT / checkpoint path." >&2
  exit 1
fi

if [[ "${RUN_TEST_UNSEEN}" == "1" ]]; then
  echo ">>> Meta-test: unseen generalization"
  python meta_test.py \
    --analysis_method unseen_generalization \
    --init_system_prompt_path "${CHECKPOINT}" \
    --log_dir "./logs/${METHOD}/unseen_generalization/${DOMAIN}" \
    "${COMMON_ARGS[@]}"
fi

if [[ "${RUN_TEST_TTA}" == "1" ]]; then
  echo ">>> Meta-test: test-time adaptation"
  python meta_test.py \
    --analysis_method test_time_adaptation \
    --init_system_prompt_path "${CHECKPOINT}" \
    --log_dir "./logs/${METHOD}/test_time_adaptation/${DOMAIN}" \
    "${COMMON_ARGS[@]}"
fi

echo "Done."
