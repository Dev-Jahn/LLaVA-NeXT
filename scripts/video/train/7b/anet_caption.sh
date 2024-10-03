#!/bin/bash
# Run the script in the project root
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

accelerate launch \
    --config_file scripts/accelerate.yaml \
    llava/train/train_voco_video.py \
    --model_path checkpoints/voco_llama_ckpt_0807 \
    --num_voco 2 \
    --conv_name vicuna_video_caption \
    --version v0 \
    --dataset activitynet-captions \
    --num_frames 32 \
    --frame_size 336 \
    --freeze_mm_mlp_adapter \
    --model_max_length 2048 \
    --bf16 True \
    --tf32 True \
    --output_dir checkpoints/voco-7b-video \
    --max_steps 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb

# Options:
#    --data_dir None \
#    --fps 1 \

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
#    --pretrain_mm_mlp_adapter checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \

#    --deepspeed scripts/zero3.json \
