from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        llm_temperature: float = 0.0,
        sequence_length: int = 4096,
        patch_unk_token: bool = False,
    ) -> None:
        super().__init__()
        self.html_tokenizer = html_tokenizer
        self.html_model = html_model
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm_prompt = llm_prompt
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.llm_temperature = llm_temperature
        self.sequence_length = sequence_length
        self.patch_unk_token = patch_unk_token
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
        if self.patch_unk_token:
            text_out_from_html = text_out_from_html.replace("<unk>", "<")
        llm_input = self.create_llm_prompt(text_out_from_html)
        # padding within batch
        input_ids = self.llm_tokenizer(llm_input, padding=True, truncation=True, return_tensors="pt")[self.sample_key]
        output = self.llm.forward(input_ids)
        return {self.prediction_key: output[self.prediction_key]}

    # Separate method so we can do something with templating later
    def create_llm_prompt(self, html_model_output: str) -> str:
        return self.llm_prompt + html_model_output

    # Ported from inference/text/inference_component.py
    def generate(self, inputs: Dict[str, torch.Tensor]):
        # remove initial pad token
        output = self.html_model.generate(inputs)[0, 1:]
        text_out_from_html = self.html_tokenizer.decode(output)
        if self.patch_unk_token:
            text_out_from_html = text_out_from_html.replace("<unk>", "<")
        llm_input = self.create_llm_prompt(text_out_from_html)
        # padding within batch
        input_ids = self.llm_tokenizer(llm_input, padding=True, truncation=True, return_tensors="pt")[self.sample_key]
        generated_token_ids = []
        generated_text_old = ""
        max_new_tokens = self.sequence_length - input_ids.shape[-1]
        for _ in range(max_new_tokens):
            logits = self.llm.forward(input_ids)["logits"]
            logits = logits[:, -1, :]
            if self.llm_temperature > 0:
                logits = logits / self.llm_temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                token_id: int = idx_next[0, 0].item()
            else:
                idx_next = torch.argmax(logits, dim=-1)
                token_id: int = idx_next.item()
            generated_token_ids.append(token_id)
            idx_next_str = self.llm_tokenizer.decode([token_id])
            generated_text_new = self.llm_tokenizer.decode(generated_token_ids)

            if idx_next_str == self.llm_tokenizer.eos_token:
                break
            else:
                diff_text = generated_text_new[len(generated_text_old) :]
                generated_text_old = generated_text_new
                print(diff_text, end="")
                input_ids = torch.cat((input_ids, torch.IntTensor([[token_id]])), dim=-1)
        return generated_text_new
