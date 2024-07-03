#!/bin/bash

CUR_DIR=/workspace/hal/AlignGPT
DATA_DIR=/workspace/hal/LLaVA/playground/data

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# N_INDICATORS=4

# FT_MODE=finetune
# FT_MODE=finetune-nogate
# FT_MODE=finetune-local
# FT_MODE=finetune-global

# INF_MODE='inference'
# INF_MODE='inference-nogate'
# INF_MODE='inference-local'
# INF_MODE='inference-global'

CKPT=aligngpt-7b
# CKPT=aligngpt-7b_ind-${N_INDICATORS}
# CKPT=aligngpt-7b_ind-${N_INDICATORS}_${FT_MODE}

ALIGN_ID=3
# OUT_DIR=${CUR_DIR}/fix_results/${SPLIT}/${CKPT}/${ALIGN_ID}/
OUT_DIR=${CUR_DIR}/fix_results/test_fix_jsonal/${CKPT}/${ALIGN_ID}/


SPLIT="llava_vqav2_mscoco_test-dev2015"
MODEL_PATH=/workspace/hal/AlignGPT/checkpoints/${CKPT} # 修改

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m src.eval.fix_vqa_loader \
        --model-path  ${MODEL_PATH} \
        --align_id ${ALIGN_ID} \
        --question-file ${CUR_DIR}/data/test_fix.jsonl \
        --image-folder ${DATA_DIR}/eval/vqav2/test2015 \
        --answers-file ${OUT_DIR}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
