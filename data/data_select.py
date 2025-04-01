import numpy as np
from datasets import load_dataset
from kcenter_greedy import SamplingMethod


class GreedyTokenSelector(SamplingMethod):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def select_batch_(self, T, mode='length'):

        token_nums = [(i, self.dataset[i]['token_num_qwen']) 
                     for i in range(self.dataset.num_rows)]
        
        if mode == 'length':
            token_nums.sort(key=lambda x: x[1], reverse=True)
        elif mode == 'sample':
            token_nums.sort(key=lambda x: x[1], reverse=False)
        else:
            raise ValueError('Sampling mode is not impletmented!')

        selected_indices = []
        current_sum = 0

        for index, token_num in token_nums:
            if current_sum >= T:
                break
            selected_indices.append(index)
            current_sum += token_num
            
        return selected_indices

if __name__ == "__main__":
    code_cot = load_dataset('efficientscaling/Z1-Code-Reasoning-107K', split='train')

    sampler = GreedyTokenSelector(code_cot)
    select_idx = sampler.select_batch_(T=73996112, mode='sample')
    ds = code_cot.select(select_idx)
    token_num = ds['token_num_qwen']
    average_token_num = np.mean(token_num)
    print(f"mean: {average_token_num}")