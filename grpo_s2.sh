#!/bin/bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-ndd-r1}"

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Omni-30B-A3B-Thinking}"
export GRPO_DATASET="${GRPO_DATASET:?Set GRPO_DATASET, e.g. /path/to/grpo_data/grpo_train.jsonl}"
export SFT_ADAPTER="${SFT_ADAPTER:?Set SFT_ADAPTER, e.g. /path/to/outputs/sft2/checkpoint-XXX}"
export OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR, e.g. /path/to/outputs/grpo2}"
export DIFFICULTY_FILE="${DIFFICULTY_FILE:?Set DIFFICULTY_FILE, e.g. /path/to/rej_sampling/clip_difficulty.json}"
export PLUGIN_PATH="${PLUGIN_PATH:-grpo_s2_plugin.py}"
export CACHE_FILE="${CACHE_FILE:?Set CACHE_FILE, e.g. /path/to/site-packages/<file containing mm_config.mm_processor_cache_gb>}"
export MASTER_PORT=$(shuf -i 30000-65000 -n 1)
export ROSCO_GRPO_DATASET="${GRPO_DATASET}"

RESUME_ARG=""
if [ -d "${OUTPUT_DIR}" ]; then
    LATEST_CKPT=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT" ]; then
        echo "RESUMING FROM: $LATEST_CKPT"
        RESUME_ARG="--resume_from_checkpoint ${LATEST_CKPT}"
    fi
fi

export USE_HF=1
export USE_AUDIO_IN_VIDEO=true
export CACHE_DIR="${CACHE_DIR:-$HOME/.cache/huggingface}"
export HF_HOME="${CACHE_DIR}"
export TRANSFORMERS_CACHE="${CACHE_DIR}"
export TORCH_HOME="${CACHE_DIR}/torch"

export ENABLE_AUDIO_OUTPUT=False
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

export MAX_PIXELS=147456
export VIDEO_MAX_PIXELS=147456
export FPS_MAX_FRAMES=2
export FPS_MIN_FRAMES=2

cp "$CACHE_FILE" "${CACHE_FILE}.bak"
sed -i 's/mm_config.mm_processor_cache_gb/max(mm_config.mm_processor_cache_gb, 10)/' "$CACHE_FILE"

mkdir -p ${OUTPUT_DIR}

GPU_LOG="${OUTPUT_DIR}/gpu_utilization_${SLURM_JOB_ID:-local}.csv"
echo "timestamp,gpu_idx,gpu_util_pct,mem_used_MiB,mem_total_MiB,power_W" > "$GPU_LOG"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits \
    -l 30 >> "$GPU_LOG" &
NVIDIA_SMI_PID=$!

cleanup() {
    kill $NVIDIA_SMI_PID 2>/dev/null || true
    cp "${CACHE_FILE}.bak" "$CACHE_FILE" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    ${RESUME_ARG} \
    --model ${MODEL_NAME} \
    --external_plugins ${PLUGIN_PATH} \
    --reward_funcs external_rosco_accuracy format \
    --reward_weights 5.0 1.0 \
    --dataset "${GRPO_DATASET}" \
    --output_dir ${OUTPUT_DIR} \
    --adapters ${SFT_ADAPTER} \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_regex '.*\.(q_proj|k_proj|v_proj|o_proj|out_proj|fc1|fc2|conv_out|qkv|proj|proj1|proj2|linear_fc1|linear_fc2)$|.*\.merger(_list\.\d+)?\.mlp\.[02]$' \
    --lora_dropout 0.05 \
    --freeze_vit false \
    --freeze_aligner false \
    --freeze_llm false \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --vllm_mm_processor_cache_gb 0 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_length 4096 \
    --num_generations 8 \
    --temperature 1.0 \
    --max_grad_norm 1.0 \
    --top_p 0.95 \
    --beta 0.001 \
    --seed 42 \
    --gradient_checkpointing true \
    --save_steps 25 \
    --save_total_limit 25 \
    --logging_steps 5 \
    --log_completions true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_enable_prefix_caching false \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --overlong_filter true \
    --dataloader_num_workers 8 \
    --add_version false
