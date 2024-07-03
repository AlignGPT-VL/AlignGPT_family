#!/bin/bash

SPLIT="mmbench_dev_20230712"

INF_MODE='inference'
# INF_MODE='inference-nogate'
# INF_MODE='inference-local'
# INF_MODE='inference-global'

python -m src.eval.model_vqa_mmbench \
    --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-7b_llama3 \
    --inference_mode ${INF_MODE} \
    --question-file /workspace/hal/LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file /workspace/hal/LLaVA/playground/data/eval/mmbench/answers/$SPLIT/aligngpt-7b_llama3.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llama_3

# mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /workspace/hal/LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir /workspace/hal/LLaVA/playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir /workspace/hal/LLaVA/playground/data/eval/mmbench/answers_upload_1/$SPLIT \
    --experiment aligngpt-7b_llama3
