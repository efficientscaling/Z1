export CUDA_VISIBLE_DEVICES=0

MODEL_DIR=/home/jovyan/workspace/ckpt/z1-coder-ckpt/z1-coder-74k
# OUTPUT_DIR=z1-ckpt/z1-coder-7b-64k/checkpoint-960


for item in "$MODEL_DIR"/*; do
    OUTPUT_DIR=$item
    echo "$item" >> "livecodebench.txt"
    python -m lcb_runner.runner.main \
        --model Qwen/CodeQwen1.5-7B-Chat \
        --model_path ${item} \
        --output_name $item \
        --scenario codegeneration \
        --max_tokens 5000 \
        --max_tokens_for_thinking 4096 \
        --n 1 \
        --temperature 0.0 \
        --top_p 1.0 \
        --stop "<|im_end|>" \
        --dtype float16 \
        --evaluate \
        --timeout 30 \
        --tensor_parallel_size 1 

    mkdir -p ${OUTPUT_DIR}
    python -m lcb_runner.runner.eval_only  \
        --generation_path ${OUTPUT_DIR}/Scenario.codegeneration_1_0.0.json \
        --scenario codegeneration \
        --output_dir ${OUTPUT_DIR} >> "livecodebench.txt"


    saved_eval_all_file=${OUTPUT_DIR}/log.json
    python -m lcb_runner.evaluation.compute_scores --eval_all_file ${saved_eval_all_file} --start_date 2024-09-01 >> "livecodebench.txt"

done


# python -m lcb_runner.runner.main \
#     --model Qwen/CodeQwen1.5-7B-Chat \
#     --model_path ${MODEL_DIR} \
#     --output_name $OUTPUT_DIR \
#     --scenario codegeneration \
#     --max_tokens 5000 \
#     --max_tokens_for_thinking 4096 \
#     --n 1 \
#     --temperature 0.0 \
#     --top_p 1.0 \
#     --stop "<|im_end|>" \
#     --dtype float16 \
#     --evaluate \
#     --tensor_parallel_size 1 

    
# mkdir -p livecodebench/evaluation/${OUTPUT_DIR}
# python -m lcb_runner.runner.eval_only  \
#     --generation_path livecodebench/output/${OUTPUT_DIR}/Scenario.codegeneration_1_0.0.json \
#     --scenario codegeneration \
#     --output_dir livecodebench/evaluation/${OUTPUT_DIR}


# saved_eval_all_file=livecodebench/evaluation/${OUTPUT_DIR}/log.json
# python -m lcb_runner.evaluation.compute_scores --eval_all_file ${saved_eval_all_file} --start_date 2024-09-01

