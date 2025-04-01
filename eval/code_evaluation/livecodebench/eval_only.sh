# saved_eval_all_file="/data/zhuotaodeng/yzj/Qwen2.5-Coder-main/qwencoder-eval/instruct/livecode_bench/evaluation/qwen15b_ins/log.json"
# OUTPUT_DIR="./evaluation/z1-coder-7b-1k"
# mkdir -p ${OUTPUT_DIR}
# python -m lcb_runner.runner.eval_only  \
#     --generation_path "/home/jovyan/workspace/ckpt/z1-ckpt/z1-coder-7b-baseline/checkpoint-480/Scenario.codegeneration_1_0.0.json" \
#     --scenario codegeneration  \
#     --output_dir ${OUTPUT_DIR}

saved_eval_all_file="/home/jovyan/workspace/ckpt/z1-coder-ckpt/z1-coder-14b-110k/checkpoint-480/log.json"
python -m lcb_runner.evaluation.compute_scores --eval_all_file ${saved_eval_all_file} --end_date 2024-05-01
