# export PATH=./bin:$PATH
# export HF_ENDPOINT=http://hf-mirror.com
# export HF_HOME=""
# export HF_DATASETS_OFFLINE=1
# export HF_EVALUATE_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
# export MODEL_DIR=${1}
# export TP=${2}
# export OUTPUT_DIR=${3}

MODEL_DIR=simplescaling/s1.1-7B

OUTPUT_DIR=/home/jovyan/workspace/z1/eval/code-evaluation/bigcodebench/outputs/simplescaling/s1.1-7B

mkdir -p ${OUTPUT_DIR}

run_benchmark() {
  SPLIT=$1
  SUBSET=$2

  # Generate code completions
  python -m bigcodebench.generate \
    --model ${MODEL_DIR} \
    --split ${SPLIT} \
    --subset ${SUBSET} \
    --greedy  \
    --bs 1 \
    --temperature 0 \
    --n_samples 1 \
    --resume  \
    --max_tokens_for_thinking 4096 \
    --max_new_tokens 5000 \
    --backend vllm \
    --tp 1 \
    --save_path ${OUTPUT_DIR}/bigcodebench_${SPLIT}_${SUBSET}/completion.jsonl \
    #--chat_mode

  # # Sanitize and calibrate the generated samples
  python -m bigcodebench.sanitize \
    --samples ${OUTPUT_DIR}/bigcodebench_${SPLIT}_${SUBSET}/completion.jsonl \
    --calibrate

  # Evaluate the sanitized and calibrated completions
  python -m bigcodebench.evaluate \
    --split ${SPLIT} \
    --subset ${SUBSET} \
    --no-gt \
    --samples ${OUTPUT_DIR}/bigcodebench_${SPLIT}_${SUBSET}/completion-sanitized-calibrated.jsonl
  
  # You are strongly recommended to use the following command to clean up the environment after evaluation:
  pids=$(ps -u $(id -u) -o pid,comm | grep 'bigcodebench' | awk '{print $1}'); if [ -n \"$pids\" ]; then echo $pids | xargs -r kill; fi;
  # rm -rf /tmp/*
}

# Run benchmarks for different configurations
# run_benchmark complete hard
# run_benchmark complete full
run_benchmark instruct hard
# run_benchmark instruct full