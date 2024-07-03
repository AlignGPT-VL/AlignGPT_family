#!/bin/bash

python -m src.eval.model_vqa_loader \
    --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-13b \
    --question-file /workspace/hal/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /workspace/hal/LLaVA/playground/data/eval/textvqa/train_images \
    --answers-file /workspace/hal/LLaVA/playground/data/eval/textvqa/answers/aligngpt-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m src.eval.eval_textvqa \
    --annotation-file /workspace/hal/LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /workspace/hal/LLaVA/playground/data/eval/textvqa/answers/aligngpt-13b.jsonl
