from abc import abstractmethod
from typing import Dict, List

import torch
import torch.nn as nn

from modalities.batch import DatasetBatch, InferenceResultBatch


class NNModel(nn.Module):
    def __init__(self, seed: int = None, optimizer_module_groups: List[str] = []):
        if seed is not None:
            torch.manual_seed(seed)
        self._optimizer_module_groups = optimizer_module_groups
        super(NNModel, self).__init__()

    @property
    def optimizer_module_groups(self):
        return self._optimizer_module_groups

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {name: param for name, param in self.named_parameters()}


def model_predict_batch(model: nn.Module, batch: DatasetBatch) -> InferenceResultBatch:
    forward_result = model.forward(batch.samples)
    result_batch = InferenceResultBatch(targets=batch.targets, predictions=forward_result)
    return result_batch
