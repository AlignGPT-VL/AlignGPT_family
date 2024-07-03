#!/bin/bash

python -m src.eval.model_vqa_loader \
    --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-13b \
    --question-file /workspace/hal/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /workspace/hal/LLaVA/playground/data/eval/pope/val2014 \
    --answers-file /workspace/hal/LLaVA/playground/data/eval/pope/answers/aligngpt-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python src/eval/eval_pope.py \
    --annotation-dir /workspace/hal/LLaVA/playground/data/eval/pope/coco \
    --question-file /workspace/hal/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /workspace/hal/LLaVA/playground/data/eval/pope/answers/aligngpt-13b.jsonl
