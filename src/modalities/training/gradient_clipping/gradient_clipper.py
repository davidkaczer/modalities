from abc import ABC, abstractmethod

import torch

from modalities.config.lookup_enum import LookupEnum


class GradientClippingMode(LookupEnum):
    P1_NORM = 1  # manhattan norm based clipping.
    P2_NORM = 2  # Euclidean norm based clipping.
    MAX_NORM = "inf"  # Maximum norm based clipping.


class GradientClipperIF(ABC):
    @abstractmethod
    def clip_gradients(self) -> torch.Tensor:
        raise NotImplementedError
