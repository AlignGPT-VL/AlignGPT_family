#!/bin/bash

python -m src.eval.model_vqa_loader \
    --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-13b \
    --question-file /workspace/hal/LLaVA/playground/data/eval/MME/llava_mme_1.jsonl \
    --image-folder /workspace/hal/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /workspace/hal/LLaVA/playground/data/eval/MME/answers/aligngpt-13b.jsonl \
    --temperature 0 \
    --conv-mode llama_3

cd /workspace/hal/LLaVA/playground/data/eval/MME

python convert_answer_to_mme.py --experiment aligngpt-13b

cd eval_tool

python calculation.py --results_dir /workspace/hal/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version/eval_tool/answers/aligngpt-13b
