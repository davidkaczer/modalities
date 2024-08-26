from typing import Optional

from pydantic import BaseModel

from modalities.config.pydanctic_if_types import PydanticPytorchModuleType, PydanticTokenizerIFType


class WebAgentLLMConfig(BaseModel):
    sample_key: str
    prediction_key: str
    llm_name: str
    html_model: PydanticPytorchModuleType
    html_tokenizer: PydanticTokenizerIFType
    llm_prompt: Optional[str]
    llm_temperature: Optional[float]
    sequence_length: Optional[int]
    patch_unk_token: Optional[bool]
