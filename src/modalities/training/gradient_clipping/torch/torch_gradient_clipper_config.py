from typing import Annotated

from pydantic import BaseModel, Field

from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.training.gradient_clipping.gradient_clipper import GradientClippingMode


class TorchGradientClipperConfig(BaseModel):
    max_norm: Annotated[float, Field(strict=True, gt=0)]
    norm_type: GradientClippingMode
    model: PydanticPytorchModuleType
