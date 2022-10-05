"""
自定义的data collators
"""
import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.data.data_collator import *
from transformers.data.data_collator import _torch_collate_batch
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask
)


class LayoutLmDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    pass
    # def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
    #     # Handle dict or lists with proper padding and conversion to tensor.
    #     if isinstance(examples[0], Mapping):
    #         batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
    #     else:
    #         batch = {
    #             "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
    #         }

    #     # If special token mask has been preprocessed, pop it from the dict.
    #     special_tokens_mask = batch.pop("special_tokens_mask", None)
    #     if self.mlm:
    #         batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
    #             batch["input_ids"], special_tokens_mask=special_tokens_mask
    #         )
    #     else:
    #         labels = batch["input_ids"].clone()
    #         if self.tokenizer.pad_token_id is not None:
    #             labels[labels == self.tokenizer.pad_token_id] = -100
    #         batch["labels"] = labels
    #     return batch
