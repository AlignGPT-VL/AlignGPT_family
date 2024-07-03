#!/bin/bash

CUR_DIR=/workspace/hal/AlignGPT
DATA_DIR=/workspace/hal/LLaVA/playground/data


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

N_INDICATORS=4

# FT_MODE=finetune
# FT_MODE=finetune-nogate
# FT_MODE=finetune-local
# FT_MODE=finetune-global

INF_MODE='inference'
# INF_MODE='inference-nogate'
# INF_MODE='inference-local'
# INF_MODE='inference-global'

# CKPT=aligngpt-7b
CKPT=aligngpt-7b_ind-${N_INDICATORS}
# CKPT=aligngpt-7b_ind-${N_INDICATORS}_${FT_MODE}



SPLIT="llava_vqav2_mscoco_test-dev2015"
MODEL_PATH=/workspace/hal/AlignGPT/checkpoints/${CKPT} # 修改

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m src.eval.model_vqa_loader \
        --model-path  ${MODEL_PATH} \
        --inference_mode ${INF_MODE} \
        --question-file ${DATA_DIR}/eval/vqav2/$SPLIT.jsonl \
        --image-folder ${DATA_DIR}/eval/vqav2/test2015 \
        --answers-file ${DATA_DIR}/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

# output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl
output_file=${DATA_DIR}/eval/vqav2/answers/$SPLIT/$CKPT/merge_1.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${DATA_DIR}/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python ${CUR_DIR}/scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT --dir ${DATA_DIR}/eval/vqav2

