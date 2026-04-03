#!/usr/bin/env bash

set -euo pipefail

export LMMS_EVAL_LAUNCHER="accelerate"
export NCCL_NVLS_ENABLE=0

# Allow overriding the key parameters via environment variables.
DEFAULT_BENCHMARKS="cvbench, blink_spatial, sparbench, videomme, mmsibench"

BENCHMARKS_RAW="${BENCHMARKS:-}"
if [[ -z "$BENCHMARKS_RAW" ]]; then
    BENCHMARKS_RAW="${BENCHMARK:-$DEFAULT_BENCHMARKS}" # choices: [vsibench, cvbench, blink_spatial, sparbench, videomme, mmsibench]
fi
IFS=',' read -ra BENCHMARK_LIST <<< "$BENCHMARKS_RAW"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/debug}"
TIMESTAMP="$(TZ="Asia/Shanghai" date "+%Y%m%d")"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_ROOT}/${TIMESTAMP}}"

mkdir -p "$OUTPUT_PATH"

NUM_MACHINES="${NUM_MACHINES:-}"
if [[ -z "$NUM_MACHINES" ]]; then
    if [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
        NUM_MACHINES="${SLURM_JOB_NUM_NODES}"
    elif [[ -n "${SLURM_NNODES:-}" ]]; then
        NUM_MACHINES="${SLURM_NNODES}"
    else
        NUM_MACHINES=1
    fi
fi

PROCESSES_PER_MACHINE="${PROCESSES_PER_MACHINE:-}"
if [[ -z "$PROCESSES_PER_MACHINE" ]]; then
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" -gt 0 ]]; then
        PROCESSES_PER_MACHINE="${SLURM_GPUS_ON_NODE}"
    elif [[ -n "${SLURM_TASKS_PER_NODE:-}" ]]; then
        PROCESSES_PER_MACHINE="${SLURM_TASKS_PER_NODE%%(*}"
    else
        if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            IFS=',' read -ra __cvd <<< "${CUDA_VISIBLE_DEVICES}"
            if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
                PROCESSES_PER_MACHINE=0
            else
                PROCESSES_PER_MACHINE="${#__cvd[@]}"
            fi
        elif command -v nvidia-smi >/dev/null 2>&1; then
            PROCESSES_PER_MACHINE="$(nvidia-smi -L | wc -l | tr -d ' ')"
        else
            PROCESSES_PER_MACHINE="$(python - <<'PY'
import os
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"
        fi
    fi
fi

if [[ -z "$PROCESSES_PER_MACHINE" || "$PROCESSES_PER_MACHINE" -lt 1 ]]; then
    PROCESSES_PER_MACHINE=1
fi

if [[ -z "$NUM_MACHINES" || "$NUM_MACHINES" -lt 1 ]]; then
    NUM_MACHINES=1
fi

TOTAL_PROCESSES=$((NUM_MACHINES * PROCESSES_PER_MACHINE))
if [[ "$TOTAL_PROCESSES" -lt 1 ]]; then
    TOTAL_PROCESSES=1
fi

MACHINE_RANK="${MACHINE_RANK:-}"
if [[ -z "$MACHINE_RANK" ]]; then
    if [[ -n "${SLURM_PROCID:-}" ]]; then
        MACHINE_RANK="${SLURM_PROCID}"
    elif [[ -n "${SLURM_NODEID:-}" ]]; then
        MACHINE_RANK="${SLURM_NODEID}"
    else
        MACHINE_RANK=0
    fi
fi

launcher_args=(--num_processes "$TOTAL_PROCESSES")
if [[ "$NUM_MACHINES" -gt 1 ]]; then
    launcher_args+=(--num_machines "$NUM_MACHINES" --machine_rank "$MACHINE_RANK")
fi
if [[ "$TOTAL_PROCESSES" -gt 1 ]]; then
    launcher_args+=(--multi_gpu)
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export MASTER_PORT="${MASTER_PORT:-29500}"
    if [[ -z "${MASTER_ADDR:-}" ]]; then
        if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_NODELIST:-}" ]]; then
            export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
        else
            export MASTER_ADDR="$(hostname)"
        fi
    fi
    launcher_args+=(--main_process_port "$MASTER_PORT")
    launcher_args+=(--main_process_ip "$MASTER_ADDR")
fi

for raw_benchmark in "${BENCHMARK_LIST[@]}"; do
    benchmark="${raw_benchmark//[[:space:]]/}"
    if [[ -z "$benchmark" ]]; then
        continue
    fi
    task_output="$OUTPUT_PATH/$benchmark"
    mkdir -p "$task_output"

    accelerate launch "${launcher_args[@]}" -m lmms_eval \
        --model spatialstack \
        --model_args pretrained="$MODEL_PATH",use_flash_attention_2=true,max_num_frames=32,max_length=12800 \
        --tasks "$benchmark" \
        --batch_size 1 \
        --output_path "$task_output" \
        --log_samples
done
