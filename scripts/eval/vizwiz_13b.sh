#!/bin/bash

python -m src.eval.model_vqa_loader \
    --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-13b \
    --question-file /workspace/hal/LLaVA/playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /workspace/hal/LLaVA/playground/data/eval/vizwiz/test \
    --answers-file /workspace/hal/LLaVA/playground/data/eval/vizwiz/answers/aligngpt-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python /workspace/hal/LLaVA/scripts/convert_vizwiz_for_submission.py \
    --annotation-file /workspace/hal/LLaVA/playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file /workspace/hal/LLaVA/playground/data/eval/vizwiz/answers/aligngpt-13b.jsonl \
    --result-upload-file /workspace/hal/LLaVA/playground/data/eval/vizwiz/answers_upload_1/aligngpt-13b.json
