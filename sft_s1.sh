#!/bin/bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-ndd-r1}"

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Omni-30B-A3B-Thinking}"
export DATASET="${DATASET:?Set DATASET, e.g. /path/to/sft_data/sft_all_balanced_shuffled.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR, e.g. /path/to/outputs/sft1}"
export MASTER_PORT=$(shuf -i 30000-65000 -n 1)

export USE_HF=1
export CACHE_DIR="${CACHE_DIR:-$HOME/.cache/huggingface}"
export HF_HOME="${CACHE_DIR}"
export HF_DATASETS_CACHE="${CACHE_DIR}"
export TRANSFORMERS_CACHE="${CACHE_DIR}"
export TORCH_HOME="${CACHE_DIR}/torch"

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

export MAX_PIXELS=147456
export VIDEO_MAX_PIXELS=147456
export FPS_MAX_FRAMES=2
export FPS_MIN_FRAMES=2
export ENABLE_AUDIO_OUTPUT=False
export USE_AUDIO_IN_VIDEO=true

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

mkdir -p ${OUTPUT_DIR}

RESUME_ARG=""
if [ -d "${OUTPUT_DIR}" ]; then
    LATEST_CKPT=$(find ${OUTPUT_DIR} -maxdepth 2 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT" ]; then
        echo "RESUMING FROM: $LATEST_CKPT"
        RESUME_ARG="--resume_from_checkpoint ${LATEST_CKPT}"
    fi
fi

GPU_LOG="${OUTPUT_DIR}/gpu_utilization_${SLURM_JOB_ID:-local}.csv"
echo "timestamp,gpu_idx,gpu_util_pct,mem_used_MiB,mem_total_MiB,power_W" > "$GPU_LOG"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits -l 30 >> "$GPU_LOG" &
NVIDIA_SMI_PID=$!
trap "kill $NVIDIA_SMI_PID 2>/dev/null" EXIT

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model ${MODEL_NAME} \
    --dataset "${DATASET}" \
    --output_dir ${OUTPUT_DIR} \
    ${RESUME_ARG} \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --target_regex '.*\.(q_proj|k_proj|v_proj|o_proj|out_proj|fc1|fc2|conv_out|qkv|proj|proj1|proj2|linear_fc1|linear_fc2)$|.*\.merger(_list\.\d+)?\.mlp\.[02]$' \
    --freeze_vit false \
    --freeze_aligner false \
    --freeze_llm false \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_length 4096 \
    --gradient_checkpointing true \
    --save_steps 10 \
    --save_total_limit 30 \
    --logging_steps 10 \
    --packing false \
    --optim adamw_torch_fused \
    --load_from_cache_file true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 16 \
    --add_version false
