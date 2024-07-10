import torch

from modalities.models.model import NNModel
from modalities.training.gradient_clipping.gradient_clipper import GradientClipperIF, GradientClippingMode


class TorchGradientClipper(GradientClipperIF):
    def __init__(self, model: NNModel, max_norm: float, norm_type=GradientClippingMode) -> None:
        self.model = model
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip_gradients(self) -> torch.Tensor:
        gradient_norm_score = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type.value
        )
        return gradient_norm_score
