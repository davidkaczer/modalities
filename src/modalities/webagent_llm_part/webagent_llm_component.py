from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from modalities.models.model import NNModel
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper


class WebAgentLLMComponent(NNModel):
    def __init__(
        self,
        llm_name: str,
        html_model: nn.Module,
        html_tokenizer: TokenizerWrapper,
        sample_key: str,
        prediction_key: str,
        llm_prompt: str = "",
    ) -> None:
        super().__init__()
        self.html_tokenizer = html_tokenizer
        self.html_model = html_model
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm_prompt = llm_prompt
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        # freeze weights of LLM
        for name, param in self.llm.named_parameters():
            param.requires_grad = False
        # known issue: LLaMA tokenizer doesn't define pad_token_id
        if self.llm_tokenizer.pad_token_id is None:
            self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # output = self.html_model.forward(inputs=inputs)
        output = self.html_model.generate(inputs)[0]
        text_out_from_html = self.html_tokenizer.decode(output)
        llm_input = self.create_llm_prompt(text_out_from_html)
        # padding within batch
        input_ids = self.llm_tokenizer(llm_input, padding=True, truncation=True, return_tensors="pt")[self.sample_key]
        output = self.llm.forward(input_ids)
        return {self.prediction_key: output[self.prediction_key]}

    def create_llm_prompt(self, html_model_output: str) -> str:
        # Separate method so we can do something with templating later
        return self.llm_prompt + html_model_output
