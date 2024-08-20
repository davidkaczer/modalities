import re
import sys
from typing import Any, Dict, List, Optional
from modalities.models.model import NNModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer


class WebAgentLLMComponent(NNModel):
    def __init__(
        self,
        llm_name: str,
        html_model: nn.Module,
        device: torch.device
    ) -> None:
        super().__init__()
        self.html_model = html_model
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.device = device
        
    def generate_tokens(
        self,
        html_model_out: str,
    ):
        # todo: implement the forward pass of the model
        pass

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass