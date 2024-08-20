from typing import Optional

from pydantic import BaseModel, field_validator

from modalities.config.pydanctic_if_types import (
    PydanticPytorchDeviceType,
    PydanticPytorchModuleType,
)
from modalities.config.utils import parse_torch_device


class WebAgentLLMConfig(BaseModel):
    llm_name: str
    html_model: PydanticPytorchModuleType
    device: PydanticPytorchDeviceType

    @field_validator("device", mode="before")
    def parse_device(cls, device) -> PydanticPytorchDeviceType:
        return parse_torch_device(device)