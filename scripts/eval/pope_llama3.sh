#!/bin/bash

INF_MODE='inference'
# INF_MODE='inference-nogate'
# INF_MODE='inference-local'
# INF_MODE='inference-global'

# python -m src.eval.model_vqa_loader \
#     --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-7b_llama3 \
#     --inference_mode ${INF_MODE} \
#     --question-file /workspace/hal/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
#     --image-folder /workspace/hal/LLaVA/playground/data/eval/pope/val2014 \
#     --answers-file /workspace/hal/LLaVA/playground/data/eval/pope/answers/aligngpt-7b_llama3.jsonl \
#     --temperature 0 \
#     --conv-mode llama_3

python src/eval/eval_pope.py \
    --annotation-dir /workspace/hal/LLaVA/playground/data/eval/pope/coco \
    --question-file /workspace/hal/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /workspace/hal/LLaVA/playground/data/eval/pope/answers/aligngpt-7b_llama3.jsonl
