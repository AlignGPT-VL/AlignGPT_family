#!/bin/bash

INF_MODE='inference'
# INF_MODE='inference-nogate'
# INF_MODE='inference-local'
# INF_MODE='inference-global'

python -m src.eval.model_vqa_loader \
    --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-7b_s2_1008 \
    --inference_mode ${INF_MODE} \
    --question-file /workspace/hal/LLaVA/playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /workspace/hal/LLaVA/playground/data/eval/vizwiz/test \
    --answers-file /workspace/hal/LLaVA/playground/data/eval/vizwiz/answers/aligngpt-7b_s2_1008.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python /workspace/hal/LLaVA/scripts/convert_vizwiz_for_submission.py \
    --annotation-file /workspace/hal/LLaVA/playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file /workspace/hal/LLaVA/playground/data/eval/vizwiz/answers/aligngpt-7b_s2_1008.jsonl \
    --result-upload-file /workspace/hal/LLaVA/playground/data/eval/vizwiz/answers_upload_1/aligngpt-7b_s2_1008.json
