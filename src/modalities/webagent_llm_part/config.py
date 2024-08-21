from typing import Optional

from pydantic import BaseModel, field_validator

from modalities.config.pydanctic_if_types import (
    PydanticPytorchModuleType,
    PydanticTokenizerIFType,
)
from modalities.config.utils import parse_torch_device


class WebAgentLLMConfig(BaseModel):
    sample_key: str
    prediction_key: str
    llm_name: str
    html_model: PydanticPytorchModuleType
    html_tokenizer: PydanticTokenizerIFType