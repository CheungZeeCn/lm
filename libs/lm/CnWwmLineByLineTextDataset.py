"""
    支持whole word masking 的dataset
        1. jieba
        2. baidu wordtag
"""
import os
import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from tqdm import tqdm

import jieba


class CnWwmLineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, engine='jieba'):
        self.examples = []

        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logging.info(f"Creating features from dataset file at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            with tqdm(total=len(lines), desc="Loading examples") as pbar:
                for line in lines:
                    chinese_ref = self.get_segments(line, engine=engine)
                    input_ids = tokenizer.encode_plus(line, add_special_tokens=True, truncation=True, max_length=block_size).input_ids
                    dict_data = {'input_ids': input_ids, 'chinese_ref': chinese_ref}
                    self.examples.append(dict_data)
                    pbar.update(1)

    def get_segments(self, input_str, engine='jieba'):
        """
            这个是按照DataCollator的要求来进行分词，按照分词结果来填充chinese_ref
        :param input_str:
        :param engine:
        :return:
        """
        chinese_ref = []
        index = 1
        if engine == 'jieba':
           seq_cws = jieba.cut(input_str)
        for seq in seq_cws:
            for i, word in enumerate(seq):
                if i > 0:
                    chinese_ref.append(index)
                index += 1
        return chinese_ref

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
