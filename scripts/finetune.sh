#!/bin/bash
LLM_DIR=/workspace/hal/llava_model
V_DIR=/workspace/hal/visual_tower
MAIN_DIR=/workspace/hal/LLaVA
DATA_DIR=${MAIN_DIR}/playground/data

CUR_DIR=./

PT_OUTPUT=aligngpt-7b-pretrain

BIN_NAME=mm_projector_align.bin

FT_OUTPUT=aligngpt-7b

# --data_path ${CUR_DIR}/data/test3.json \
# --image_folder ${DATA_DIR}/LLaVA-Pretrain/images \


# --data_path ${MAIN_DIR}/playground/data/llava_v1_5_mix665k.json \
# --image_folder ${MAIN_DIR}/playground/data \


# --pretrain_mm_mlp_adapter ${CUR_DIR}/checkpoints/${PT_OUTPUT}/${BIN_NAME} \

deepspeed --include localhost:1,5,6,7 --master_port=30000 ${CUR_DIR}/src/train/train_mem_flash.py \
    --deepspeed ${MAIN_DIR}/scripts/zero3.json \
    --model_name_or_path ${LLM_DIR}/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ${MAIN_DIR}/playground/data/llava_v1_5_mix665k.json \
    --image_folder ${MAIN_DIR}/playground/data \
    --vision_tower  ${V_DIR}/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_align ${CUR_DIR}/checkpoints/${PT_OUTPUT}/${BIN_NAME} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${CUR_DIR}/checkpoints/${FT_OUTPUT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --stage finetune \