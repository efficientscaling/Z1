#!/bin/bash

# 环境变量设置
export PATH=./bin:$PATH
export HF_ENDPOINT=http://hf-mirror.com
export HF_HOME=""
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

# 定义检查点路径和输出路径
MODEL_DIR_1=/home/jovyan/workspace/ckpt/z1-ckpt/z1-math-7b-1k/checkpoint-60
MODEL_DIR_2=/home/jovyan/workspace/ckpt/z1-ckpt/z1-math-7b-1k/checkpoint-120

OUTPUT_DIR_1=/home/jovyan/workspace/z1/eval/BigCodeBench/results/BigCodeBench_z1-math-7b-1k-60
OUTPUT_DIR_2=/home/jovyan/workspace/z1/eval/BigCodeBench/results/BigCodeBench_z1-math-7b-1k-120

# 创建输出目录
mkdir -p ${OUTPUT_DIR_1}
mkdir -p ${OUTPUT_DIR_2}



run_benchmark() {
  GPU_ID=$1
  MODEL_DIR=$2
  OUTPUT_DIR=$3
  SPLIT=$4
  SUBSET=$5

  # 设置 GPU
  export CUDA_VISIBLE_DEVICES=${GPU_ID}

  # 生成代码补全
  python generate.py \
    --model ${MODEL_DIR} \
    --split ${SPLIT} \
    --subset ${SUBSET} \
    --greedy  \
    --bs 1 \
    --temperature 0 \
    --n_samples 1 \
    --resume  \
    --backend vllm \
    --tp 1 \
    --save_path ${OUTPUT_DIR}/bigcodebench_${SPLIT}_${SUBSET}/completion.jsonl \
    #--chat_mode

  # 清理和校准生成的样本
  python sanitize.py \
    --samples ${OUTPUT_DIR}/bigcodebench_${SPLIT}_${SUBSET}/completion.jsonl \
    --calibrate

  # 评估清理和校准后的补全
  python evaluate.py \
    --split ${SPLIT} \
    --subset ${SUBSET} \
    --no-gt \
    --samples ${OUTPUT_DIR}/bigcodebench_${SPLIT}_${SUBSET}/completion-sanitized-calibrated.jsonl
  
  # 清理环境
  pids=$(ps -u $(id -u) -o pid,comm | grep 'bigcodebench' | awk '{print $1}'); if [ -n "$pids" ]; then echo $pids | xargs -r kill; fi;
  # rm -rf /tmp/*
}

# 使用不同的 GPU 和检查点运行基准测试
run_benchmark 0 ${MODEL_DIR_1} ${OUTPUT_DIR_1} complete hard &
run_benchmark 1 ${MODEL_DIR_2} ${OUTPUT_DIR_2} complete hard &

run_benchmark 0 ${MODEL_DIR_1} ${OUTPUT_DIR_1} instruct hard &
run_benchmark 1 ${MODEL_DIR_2} ${OUTPUT_DIR_2} instruct hard &
# 等待所有任务完成
wait
