#!/bin/bash

# if [ "$1" = "dev" ]; then
#     ZH_SPLIT="验证集"
#     echo "Evaluating in 'dev' split."
# elif [ "$1" = "test" ]; then
#     ZH_SPLIT="测试集"
#     echo "Evaluating in 'test' split."
# else
#     echo "Unknown split, please choose between 'dev' and 'test'."
#     exit 1
# fi

python -m llava.eval.model_vqa_qbench \
    --model-path /workspace/hal/checkpoints/llava-v1.5-7b-finetune-align-A800-8-8-4-8-16-1-imprv \
    --image-folder /workspace/hal/LLaVA/playground/data/eval/qbench_zh/images/ \
    --questions-file /workspace/hal/LLaVA/playground/data/eval/qbench_zh/质衡-问答-验证集.json \
    --answers-file /workspace/hal/LLaVA/playground/data/eval/qbench_zh/llava-v1.5-7b-finetune-align-A800-8-8-4-8-16-1-imprv.jsonl \
    --conv-mode llava_v1 \
    --lang zh

python /workspace/hal/LLaVA/playground/data/eval/qbench_zh/format_qbench.py \
    --filepath /workspace/hal/LLaVA/playground/data/eval/qbench_zh/llava-v1.5-7b-finetune-align-A800-8-8-4-8-16-1-imprv.jsonl

python /workspace/hal/LLaVA/playground/data/eval/qbench_zh/qbench_eval.py \
    --filepath /workspace/hal/LLaVA/playground/data/eval/qbench_zh/llava-v1.5-7b-finetune-align-A800-8-8-4-8-16-1-imprv.jsonl
