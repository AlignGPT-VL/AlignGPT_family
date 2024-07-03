#!/bin/bash

python -m src.eval.model_vqa \
    --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-7b \
    --image-folder /workspace/hal/LLaVA/playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file /workspace/hal/LLaVA/playground/data/eval/llava-bench-in-the-wild/answers/aligngpt-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# mkdir -p /workspace/hal/LLaVA/playground/data/eval/llava-bench-in-the-wild/reviews

python /workspace/hal/LLaVA/llava/eval/eval_gpt_review_bench.py \
    --question /workspace/hal/LLaVA/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context /workspace/hal/LLaVA/playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule /workspace/hal/LLaVA/llava/eval/table/rule.json \
    --answer-list \
        /workspace/hal/LLaVA/playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        /workspace/hal/LLaVA/playground/data/eval/llava-bench-in-the-wild/answers/aligngpt-7b.jsonl \
    --output \
        /workspace/hal/LLaVA/playground/data/eval/llava-bench-in-the-wild/reviews/aligngpt-7b.jsonl

python /workspace/hal/LLaVA/llava/eval/summarize_gpt_review.py -f /workspace/hal/LLaVA/playground/data/eval/llava-bench-in-the-wild/reviews/aligngpt-7b.jsonl
