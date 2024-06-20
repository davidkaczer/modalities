from dataclasses import field
from typing import Dict, List

import torch
from pydantic import BaseModel

from modalities.batch import DatasetBatch
from modalities.models.gpt2.collator import CollateFnIF


class AnyMALCollateFnConfig(BaseModel):
    sample_keys: List[str]
    text_sample_key: str
    text_target_key: str
    n_modality_tokens: int


class AnyMALCollatorFn(CollateFnIF):
    def __init__(self, sample_keys: List[str], text_sample_key: str, text_target_key: str, n_modality_tokens: int):
        self.device: torch.device = field(default_factory=lambda: torch.device("cpu"))
        if text_sample_key not in sample_keys:
            raise ValueError(f"{text_sample_key} is not part of sample keys {sample_keys}")
        self.sample_keys = sample_keys  # e.g. ['images', 'input_ids']
        self.text_sample_key = text_sample_key  # input_ids
        self.text_target_key = text_target_key  # target_ids
        self.n_modality_tokens = n_modality_tokens

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch:
        samples = {
            sample_key: torch.stack([torch.tensor(d[sample_key]) for d in batch]) for sample_key in self.sample_keys
        }

        # Create target for text input
        targets = {}
        targets[self.text_target_key] = samples[self.text_sample_key][:, 1:].clone().detach()
        B, L = targets[self.text_target_key].size()

        # prepend `ignore_index` equal to n_modality_tokens
        # see ignore_index: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        ignore = torch.ones(B, self.n_modality_tokens, dtype=targets[self.text_target_key].dtype) * (-100)
        targets[self.text_target_key] = torch.cat((ignore, targets[self.text_target_key]), axis=1)

        samples[self.text_sample_key] = samples[self.text_sample_key][:, :-1].clone().detach()
        return DatasetBatch(targets=targets, samples=samples)
