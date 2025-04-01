MODEL_DIR=/home/jovyan/workspace/ckpt/z1-ckpt/z1-coder-7b-baseline/checkpoint-480

CUDA_VISIBLE_DEVICES='0' \
python eval.py \
    --model_name_or_path $MODEL_DIR \
    --data_name gpqa \
    --prompt_type "qwen-instruct" \
    --temperature 0.0 \
    --start_idx 0 \
    --end_idx -1 \
    --n_sampling 1 \
    --k 1 \
    --split "test" \
    --max_thinking_tokens 2048 \
    --max_tokens 3200 \
    --seed 0 \
    --top_p 1 \
    --surround_with_messages 