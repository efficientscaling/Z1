import copy
from typing import List
from dataclasses import dataclass
from vllm import LLM, SamplingParams
from transformers import  AutoTokenizer

BOX=r"\boxed{}"
ANSWER_WITH_BOX=f"\n\nI overthought it, the final answer in {BOX} should be:\n\n"
ANSWER_WITHOUT_BOX=f"\n\nI overthought it, the final answer should be:\n\n"

@dataclass
class ThinkingLLM(LLM):

    def __init__(self, *args, **kwargs):
        """
        Initialize the ThinkingLLM class.

        Args:
            max_tokens_thinking (int): Maximum budget in terms of tokens.
            *args, **kwargs: Additional arguments passed to the parent LLM class.
        """
        super().__init__(*args, **kwargs)

    def thinking_generate(self, prompts: List[str], sampling_params: SamplingParams = None, max_tokens_for_thinking: int = None):
        """
        Generate text with a specified budget.

        Args:
            prompt (str): The input prompt for the LLM.
            sampling_params (SamplingParams): A SamplingParams object to configure generation.
            budget (int): The maximum budget for generation (e.g., token limit).
                          If None, defaults to the instance's max_budget.

        Returns:
            str: The generated text within the budget.
        """

        # If no SamplingParams is provided, create a default one
        if sampling_params is None:
            raise ValueError("Sampling_params can't be None!")
        else:
            all_max_tokens = sampling_params.max_tokens
            # Override the max_tokens in the provided SamplingParams with the budget
            sampling_params.max_tokens = max_tokens_for_thinking
            print(f"All tokens: {all_max_tokens}")
            print(f"Tokens for thinking: {max_tokens_for_thinking}")

        trajectories = self.generate(prompts, sampling_params)

        rethinking_str = ANSWER_WITHOUT_BOX
        sampling_params.max_tokens = all_max_tokens

        answers = copy.deepcopy(trajectories)

        unfinished_id = []
        thinking_token = 0
        new_prompts = []

        for id, traj in enumerate(trajectories):
            if traj.outputs[0].finish_reason == 'length':
                unfinished_id.append(id)
                new_prompts.append(prompts[id] + traj.outputs[0].text + rethinking_str)
            thinking_token += len(traj.outputs[0].token_ids)

        avg_thinking_token = thinking_token / len(prompts)

        if new_prompts:
            print(new_prompts[0])

            o = self.generate(
                new_prompts,
                sampling_params=sampling_params,
            )
            
        for i, uid in enumerate(unfinished_id):
            answers[uid] = o[i]

        return avg_thinking_token, answers

if __name__ == '__main__':

    thinking_llm = ThinkingLLM(
        model='efficientscaling/Z1-7B',
        tensor_parallel_size=1,
        gpu_memory_utilization=0.96,
    )
    tokenizer = AutoTokenizer.from_pretrained('efficientscaling/Z1-7B')
    stop_token_ids = tokenizer("<|im_start|><|im_end|>")["input_ids"]

    prompts = [
        "Who are you?",
        "Let $p(x)$ be a polynomial of degree 5 such that \[p(n) = \frac{n}{n^2 - 1}\]for $n = 2,$ 3, 4, $\dots,$ 7. Find $p(8).$",
    ]

    prompts = ["<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n" for p in prompts]

    sampling_params = SamplingParams(
    skip_special_tokens=False,
    temperature=0.0,
    max_tokens=8000,
    stop_token_ids=stop_token_ids
    )
        
    _, o = thinking_llm.thinking_generate(
        prompts,
        sampling_params=sampling_params,
        max_tokens_for_thinking=4096
    )

    print(o[0].outputs[0].text)