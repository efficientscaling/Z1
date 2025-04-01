# export CUDA_VISIBLE_DEVICES=2,5,6
export WANDB_DISABLED=true
set -ex

SAVE_PATH=/home/jovyan/workspace/ckpt/z1-coder-ckpt/z1-coder-74k
mkdir -p $SAVE_PATH

torchrun --nproc_per_node=8 --master_port=20011 train.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
    --data_path efficientscaling/Z1-Code-Reasoning-107K \
    --bf16 True \
    --tf32 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 120 \
    --save_total_limit 17 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config /home/jovyan/workspace/z1/train/fsdp_config_qwen_cpu.json \
    --gradient_checkpointing True 

     

