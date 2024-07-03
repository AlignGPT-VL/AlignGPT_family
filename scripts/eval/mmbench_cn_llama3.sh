#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"

INF_MODE='inference'
# INF_MODE='inference-nogate'
# INF_MODE='inference-local'
# INF_MODE='inference-global'

python -m src.eval.model_vqa_mmbench \
    --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-7b_llama3 \
    --inference_mode ${INF_MODE} \
    --question-file /workspace/hal/LLaVA/playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file /workspace/hal/LLaVA/playground/data/eval/mmbench_cn/answers/$SPLIT/aligngpt-7b_llama3.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llama_3

# mkdir -p /workspace/hal/LLaVA/playground/data/eval/mmbench_cn/answers_upload_1/$SPLIT

python /workspace/hal/LLaVA/scripts/convert_mmbench_for_submission.py \
    --annotation-file /workspace/hal/LLaVA/playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir /workspace/hal/LLaVA/playground/data/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir /workspace/hal/LLaVA/playground/data/eval/mmbench_cn/answers_upload_1/$SPLIT \
    --experiment aligngpt-7b_llama3
