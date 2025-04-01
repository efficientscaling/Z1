<h1 align="center">
<img src="./assets/logo.png" width="120" alt="Z1-Coder" />
<br>
Z1: Efficient Test-time Scaling with Code


</h1>

<p align="center">
  <a href=""><b>[📜 Paper]</b></a> •
  <a href="https://huggingface.co/efficientscaling/Z1-7B"><b>[🤗 HF Models]</b></a> •
  <a href="https://github.com/efficientscaling/Z1"><b>[🐱 GitHub]</b></a>
  <!-- <a href="https://9557c5365a6f44dc84.gradio.live"><b>[🐯 Gradio Demo]</b></a> -->
  <br>

  <!-- <a href="#-quick-start">Quick Start</a> • -->
  <!-- <a href="#%EF%B8%8F-citation">Citation</a> -->
</p>

<p align="center">
    <img src="./assets/z1-comp.png" width="800">
    <img src="./assets/tts.png" width="800">
    <br>

</p>
<!-- <div align="center">

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#updates" style="text-decoration: none; font-weight: bold;">Updates</a> •
    <a href="#links" style="text-decoration: none; font-weight: bold;">Links</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">Getting Started</a> •
    <a href="#introduction" style="text-decoration: none; font-weight: bold;">Introduction</a> •
    <a href="#evaluation" style="text-decoration: none; font-weight: bold;">Evaluation</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">Citation</a> 
  </p>
</div>

</div> -->


## Updates
<!-- - **[2025/01/17]** 🎉 We have released our [Blog]([https://huggingface.co/Z1-Coder](https://z1-coder.github.io/Z1CODER/posts/202501-z1-coder-init/z1-coder-blog/))! -->
- **[2025/04/01]** 🎉 We have released our [Z1-7B](https://huggingface.co/efficientscaling/Z1-7B) model and  [data](https://huggingface.co/datasets/efficientscaling/Z1-Code-Reasoning-107K) through Huggingface!


<!-- # Links

- 🤗 [Z1 models](https://huggingface.co/efficientscaling/Z1-7B)
- 🤗 [Z1 data](https://huggingface.co/datasets/efficientscaling/Z1-Code-Reasoning-107K) -->

## Getting Started

We open source the code and scripts we used for data curation, training, and evaluation for Z1 models, you can find more details in each directory.

- ``src/eval``:  Evaluation scripts for Z1.
- ``src/data``:  Data abaltion scripts for Z1.
- ``scr/train``: Training scripts for Z1. We train Z1-7B with Fully Shard Data Parallel (FSDP) and set a global batch size to 128 for 2 epochs using 8 NVIDIA A100-80G GPUs.

## Inference

#### vLLM with Shifted Thinking Window

Install `vllm` library firstly. We use `vllm==0.5.3.post1`.

```python
import copy
from typing import List
from dataclasses import dataclass
from code_evaluation.reasoning import ThinkingLLM
from transformers import  AutoTokenizer

thinking_llm = ThinkingLLM(
    model='efficientscaling/Z1-7B',
    tensor_parallel_size=1,
    gpu_memory_utilization=0.96,
)
tokenizer = AutoTokenizer.from_pretrained('efficientscaling/Z1-7B')
stop_token_ids = tokenizer("<|im_end|>")["input_ids"]

prompts = [
    "Write a Python script to calculate the number of letter ’a’ and ’r’ in a string.",
]

prompts = ["<|im_start|>system\nPlease reason step by step.<|im_end|>\n<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n" for p in prompts]

sampling_params = SamplingParams(
skip_special_tokens=False,
temperature=0.0,
top_p=1,
max_tokens=5000,
# stop_token_ids=stop_token_ids,
)
    
avg_thinking_tokens, o = thinking_llm.thinking_generate(
    prompts,
    sampling_params=sampling_params,
    max_tokens_for_thinking=4096
)

print(o[0].outputs[0])
print(avg_thinking_tokens)

```

## Training
If the dataset is too large, you can first find our script at `train/preprocess.py` to tokenize and save your dataset if the dataset too large.
Then run `train/script/train_qwen.sh`  launch training.

We train Z1-7B with Fully Shard Data Parallel (FSDP) and set a global batch size to 128 for 2 epochs using 8 NVIDIA A100-80G GPUs. You can downgrade the `max_tokens` in your training dataset to avoid OOM.

## Evaluation

We clone [LIMO](https://github.com/GAIR-NLP/LIMO) repository and modift its evaluation scipt to evaluate Z1 on MATH500 and GPQA.
Environment Setup:
```sh
cd eval/general_evaluation
pip install -r 'requirements.txt'
```
For LiveCodeBench and BigCodeBench, you can install the environment with following commands:
```sh
cd eval/code_evaluation/livecodebench
pip install -e .
```
and 
```sh
cd eval/code_evaluation/bigcodebench
pip install -r 'requirements.txt'
```
