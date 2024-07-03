#!/bin/bash

LLM_DIR=/workspace/hal/llava_model
V_DIR=/workspace/hal/visual_tower
MAIN_DIR=/workspace/hal/LLaVA
DATA_DIR=${MAIN_DIR}/playground/data

CUR_DIR=./

N_INDICATORS=8

PT_OUTPUT=aligngpt-7b-pretrain_ind-${N_INDICATORS}
BIN_NAME=mm_projector_align.bin
FT_OUTPUT=aligngpt-7b_ind-${N_INDICATORS}

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=30001 ${CUR_DIR}/src/train/train_mem_flash.py \
    --deepspeed ${CUR_DIR}/scripts/zero2.json \
    --model_name_or_path ${LLM_DIR}/vicuna-7b-v1.5 \
    --version plain \
    --n_indicators ${N_INDICATORS} \
    --data_path ${MAIN_DIR}/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k_with_similarity_number_${N_INDICATORS}.json \
    --image_folder ${DATA_DIR}/LLaVA-Pretrain/images \
    --vision_tower ${V_DIR}/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${CUR_DIR}/checkpoints/${PT_OUTPUT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4\
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --stage pretrain

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=30002 ${CUR_DIR}/src/train/train_mem_flash.py \
    --deepspeed ${MAIN_DIR}/scripts/zero3.json \
    --model_name_or_path ${LLM_DIR}/vicuna-7b-v1.5 \
    --version v1 \
    --n_indicators ${N_INDICATORS} \
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
    --gradient_accumulation_steps 1 \
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
    --report_to wandb \
    --stage finetune \
