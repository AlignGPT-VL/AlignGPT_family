#!/bin/bash

INF_MODE='inference'
# INF_MODE='inference-nogate'
# INF_MODE='inference-local'
# INF_MODE='inference-global'

python -m src.eval.model_vqa_loader \
    --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-7b_s2_1008 \
    --inference_mode ${INF_MODE} \
    --question-file /workspace/hal/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /workspace/hal/LLaVA/playground/data/eval/textvqa/train_images \
    --answers-file /workspace/hal/LLaVA/playground/data/eval/textvqa/answers/aligngpt-7b_s2_1008.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m src.eval.eval_textvqa \
    --annotation-file /workspace/hal/LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /workspace/hal/LLaVA/playground/data/eval/textvqa/answers/aligngpt-7b_s2_1008.jsonl
