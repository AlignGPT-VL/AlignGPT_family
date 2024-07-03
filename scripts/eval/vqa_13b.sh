#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="aligngpt-13b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m src.eval.model_vqa_loader \
        --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-13b \
        --question-file /workspace/hal/LLaVA/playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder /workspace/hal/LLaVA/playground/data/eval/vqav2/test2015 \
        --answers-file /workspace/hal/LLaVA/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

# output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl
output_file=/workspace/hal/LLaVA/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge_1.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /workspace/hal/LLaVA/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python /workspace/hal/LLaVA/scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT