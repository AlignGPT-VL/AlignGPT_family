#!/bin/bash

INF_MODE='inference'
# INF_MODE='inference-nogate'
# INF_MODE='inference-local'
# INF_MODE='inference-global'

python -m src.eval.model_vqa_loader \
    --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-7b \
    --inference_mode ${INF_MODE} \
    --question-file /workspace/hal/LLaVA/playground/data/eval/MME/llava_mme_1.jsonl \
    --image-folder /workspace/hal/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /workspace/hal/LLaVA/playground/data/eval/MME/answers/aligngpt-7b_s2_1008.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /workspace/hal/LLaVA/playground/data/eval/MME

python convert_answer_to_mme.py --experiment aligngpt-7b_s2_1008

cd eval_tool

python calculation.py --results_dir /workspace/hal/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version/eval_tool/answers/aligngpt-7b_s2_1008
