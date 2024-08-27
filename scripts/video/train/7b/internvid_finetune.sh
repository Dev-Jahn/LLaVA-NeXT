#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch llava.train.train_voco_video.py \
    --model_path checkpoints checkpoints/voco_llama_ckpt_0807 \
    --pretrain_mm_mlp_adapter checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --version v0 \
    --mm_vision_select_layer -2 \
    --dataset internvid \
    --data_dir None \
    --fps None \
    --num_frames 32 \
    --frame_size 336 \
    --cache_dir: None \
    --model_max_length 2048

#    --freeze_backbone \
#    --tune_mm_mlp_adapter \
#    --group_by_modality_length \

#    --lora_enable \
#    --lora_r 64 \
#    --lora_alpha 16 \
#    --lora_dropout 0.05 \
#    --lora_weight_path "" \
#    --lora_bias "none" \
#    --mm_projector_lr None \
