from dataclasses import dataclass, field
import logging
import pathlib
import copy
from typing import Dict, Optional, Sequence
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import utils
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = load_dataset(data_path, split='train')
        logging.warning("Formatting inputs...")
        sources = [
            (
                prompt_format(tokenizer, example)
            )
            for example in list_data_dict
        ]
        targets = [
            f"{example['response']}{'<|im_end|>'}" for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    

def save_dataset(dataset: SupervisedDataset, save_path: str):

    save_dir = pathlib.Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(dataset.input_ids, save_dir / "input_ids.pt")
    torch.save(dataset.labels, save_dir / "labels.pt")
    logging.info(f"Dataset saved to {save_dir}")


def load_from_pt(save_path: str) -> SupervisedDataset:

    save_dir = pathlib.Path(save_path)

    # 加载 input_ids 和 labels
    input_ids = torch.load(save_dir / "input_ids.pt")
    labels = torch.load(save_dir / "labels.pt")

    # 创建一个空的 SupervisedDataset 实例
    dataset = SupervisedDataset.__new__(SupervisedDataset)
    dataset.input_ids = input_ids
    dataset.labels = labels

    logging.info(f"Dataset loaded from {save_dir}")
    return dataset


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (tqdm(examples), sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print(f"len={len(train_dataset)}")
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def prompt_format(tokenizer, example):
    question = example['question'].strip()
    messages = [
        {
            "role": "user",
            "content": question
        },
        {
            "role": "assistant",
            "content": '' + _MAGIC_SPLITTER_            
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False).split(_MAGIC_SPLITTER_)[0]

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('/work/zhuotaodeng/yzj/pretrained_models_ms/Qwen/Qwen2___5-7B-Instruct')
    # data_path = '/data/zhuotaodeng/test-time-scaling/z1/data/openthought_evol-221k.json'
    # ds = make_supervised_data_module(tokenizer, data_path)
    # save_dataset(ds['train_dataset'],'/data/zhuotaodeng/test-time-scaling/z1/data/qwen')

    ds = load_dataset('efficientscaling/Z1-Code-Reasoning-107K')
    data = ds[1]
    decoded_input = tokenizer.decode(data['input_ids'], skip_special_tokens=True)
    print("Decoded input_ids:", decoded_input)
    filtered_labels = data['labels'][data['labels'] != -100]
    decoded_labels = tokenizer.decode(filtered_labels, skip_special_tokens=True)
    print("Decoded labels:", decoded_labels)
