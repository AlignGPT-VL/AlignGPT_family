#!/bin/bash

INF_MODE='inference'
# INF_MODE='inference-nogate'
# INF_MODE='inference-local'
# INF_MODE='inference-global'

# python -m src.eval.model_vqa_science \
#     --model-path /workspace/hal/AlignGPT/checkpoints/aligngpt-7b_llama3 \
#     --inference_mode ${INF_MODE} \
#     --question-file /workspace/hal/LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
#     --image-folder /workspace/hal/LLaVA/playground/data/eval/scienceqa/ScienceQA_DATA/test \
#     --answers-file /workspace/hal/LLaVA/playground/data/eval/scienceqa/answers/aligngpt-7b_llama3.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode llama_3

python src/eval/eval_science_qa.py \
    --base-dir /workspace/hal/LLaVA/playground/data/eval/scienceqa/ScienceQA_DATA \
    --result-file /workspace/hal/LLaVA/playground/data/eval/scienceqa/answers/aligngpt-7b_llama3.jsonl \
    --output-file /workspace/hal/LLaVA/playground/data/eval/scienceqa/answers/aligngpt-7b_llama3_output.jsonl \
    --output-result /workspace/hal/LLaVA/playground/data/eval/scienceqa/answers/aligngpt-7b_llama3_result.json
