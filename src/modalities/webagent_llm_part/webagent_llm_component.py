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
        html_tokenizer: TokenizerWrapper,
        sample_key: str,
        prediction_key: str,
    ) -> None:
        super().__init__()
        self.html_tokenizer = html_tokenizer
        self.html_model = html_model
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        for name, param in self.llm.named_parameters():
            param.requires_grad = False

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.html_model.forward(inputs[self.sample_key])
        text_out_from_html = self.html_tokenizer.batch_decode(output[self.prediction_key])
        # padding within batch
        input_ids = self.llm_tokenizer(text_out_from_html, padding=True, truncation=True, return_tensors="pt")[self.sample_key]
        output = self.llm.forward(input_ids)
        return {self.prediction_key: output[self.prediction_key]}