#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

INF_MODE='inference'
# INF_MODE='inference-nogate'
# INF_MODE='inference-local'
# INF_MODE='inference-global'

CKPT="aligngpt-7b_llama3"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="/workspace/hal/LLaVA/playground/data/eval/gqa/data"

# --inference_mode ${INF_MODE} \

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m src.eval.model_vqa_loader \
#         --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-7b_llama3 \
#         --inference_mode ${INF_MODE} \
#         --question-file /workspace/hal/LLaVA/playground/data/eval/gqa/$SPLIT.jsonl \
#         --image-folder /workspace/hal/LLaVA/playground/data/eval/gqa/data/images \
#         --answers-file /workspace/hal/LLaVA/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode llama_3 &
# done

# wait

# output_file=/workspace/hal/LLaVA/playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl
output_file=/workspace/hal/LLaVA/playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge_1.jsonl

# Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat /workspace/hal/LLaVA/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

python /workspace/hal/LLaVA/scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
# python eval/eval.py --tier testdev_balanced
python eval/eval_1.py --tier testdev_balanced