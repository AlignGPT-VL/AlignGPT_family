#!/bin/bash

python -m src.eval.model_vqa \
    --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-7b \
    --question-file /workspace/hal/LLaVA/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /workspace/hal/LLaVA/playground/data/eval/mm-vet/mm-vet/images \
    --answers-file /workspace/hal/LLaVA/playground/data/eval/mm-vet/answers/aligngpt-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# mkdir -p ./playground/data/eval/mm-vet/results

python /workspace/hal/LLaVA/scripts/convert_mmvet_for_eval.py \
    --src /workspace/hal/LLaVA/playground/data/eval/mm-vet/answers/aligngpt-7b.jsonl \
    --dst /workspace/hal/LLaVA/playground/data/eval/mm-vet/results/aligngpt-7b.json

