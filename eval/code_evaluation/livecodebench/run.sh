export CUDA_VISIBLE_DEVICES=0
MODEL_DIR=/home/jovyan/workspace/ckpt/z1-ckpt/z1-coder-7b-baseline/checkpoint-720
# MODEL_DIR=/home/jovyan/.cache/huggingface/hub/models--zjy2001--z1-coder-7b-4k-c120/snapshots/a8f976e95b2560f635bc9950cdfa4fc132ffb6e9

OUTPUT_DIR=z1-ckpt/z1-coder-7b-baseline/checkpoint-480
# OUTPUT_DIR=z1-coder-ckpt/models--zjy2001--z1-coder-7b-4k-c120
echo "$OUTPUT_DIR" >> "token.txt"
python -m lcb_runner.runner.main \
    --model Qwen/CodeQwen1.5-7B-Chat \
    --model_path ${MODEL_DIR} \
    --output_name $OUTPUT_DIR \
    --scenario codegeneration \
    --max_tokens 3000 \
    --max_tokens_for_thinking 2048 \
    --n 1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --stop "<|im_end|>" \
    --dtype float16 \
    --evaluate \
    --tensor_parallel_size 1 

    
mkdir -p ${OUTPUT_DIR}
python -m lcb_runner.runner.eval_only  \
    --generation_path ${OUTPUT_DIR}/Scenario.codegeneration_1_0.0.json \
    --scenario codegeneration \
    --output_dir ${OUTPUT_DIR}


saved_eval_all_file=${OUTPUT_DIR}/log.json
python -m lcb_runner.evaluation.compute_scores --eval_all_file ${saved_eval_all_file} --start_date 2024-09-01

