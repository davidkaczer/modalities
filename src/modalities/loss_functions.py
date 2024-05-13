from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import BaseModel
from torch.nn import CrossEntropyLoss as TorchCrossEntropyLoss

from modalities.batch import InferenceResultBatch


class Loss(ABC):
    def __init__(self, tag: str, weight: float = 1.0):
        self._tag = tag
        self.weight = weight

    @property
    def tag(self) -> str:
        return self._tag

    @abstractmethod
    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Calculates the loss
        :return: Loss tensor
        """
        raise NotImplementedError


class CrossEntropyLossConfig(BaseModel):
    target_key: str
    prediction_key: str
    weight: float = 1
    tag: str = "CLMCrossEntropyLoss"


class CrossEntropyLoss(Loss):
    def __init__(self, target_key: str, prediction_key: str, weight: float, tag: str = "CLMCrossEntropyLoss"):
        super().__init__(tag, weight)
        self.target_key = target_key
        self.prediction_key = prediction_key
        # Mean over the tokens in the local-batch (batch per rank)
        self.loss_fun = TorchCrossEntropyLoss(reduction="mean")

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        labels = forward_batch.get_targets(self.target_key)

        if "attention_mask" in forward_batch.targets:
            attention_mask = forward_batch.get_targets("attention_mask")
            labels[attention_mask == 0] = -100

        lm_logits = forward_batch.get_predictions(self.prediction_key)

        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits.contiguous()
        shift_labels = labels.contiguous().long()
        # Flatten the tokens. We compute here, the loss per token.
        loss = self.loss_fun(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss


def nce_loss(
    embedding1: torch.Tensor, embedding2: torch.Tensor, device: torch.device, is_asymmetric: bool, temperature: float
) -> torch.Tensor:
    """
    This implementation calculates the noise contrastive estimation loss between embeddings of two different modalities
    Implementation slightly adapted from https://arxiv.org/pdf/1912.06430.pdf, https://github.com/antoine77340/MIL-NCE_HowTo100M
    changes include adding a temperature value and the choice of calculating asymmetric loss w.r.t. one modality
    This implementation is adapted to contrastive loss from CoCa model https://arxiv.org/pdf/2205.01917.pdf

    Args:
        embedding1 (torch.Tensor): embeddings from modality 1 of size batch_size x embed_dim.
        embedding2 (torch.Tensor): embeddings from modality 2 of size batch_size x embed_dim.
        device (torch.device): torch device for calculating loss.
        is_asymmetric (bool): boolean value to specify if the loss is calculated in one direction or both directions.
        temperature (float): temperature value for regulating loss.

    Returns:
            torch.Tensor: loss tensor.
    """
    # calculating the similarity matrix of size (batch_size x batch_size)
    sim_matrix = torch.matmul(embedding1, embedding2.t()) / temperature
    # numerator of loss: using similarity scores for all positive pairs (e.g., image and its caption)
    numerator = sim_matrix * torch.eye(sim_matrix.shape[0], device=device)
    numerator = numerator.sum(dim=0).view(sim_matrix.shape[0], -1)
    numerator = torch.logsumexp(numerator, dim=1)
    if is_asymmetric:
        # denominator of loss: using all similarity scores for all pairs (positive and negative)
        denominator = torch.logsumexp(sim_matrix, dim=1)
    else:
        # calculate bidirectional loss
        numerator *= 2
        denominator = torch.logsumexp(sim_matrix, dim=1) + torch.logsumexp(sim_matrix.t(), dim=1)
    return torch.mean(denominator - numerator)  # calculated in log space


class NCELossConfig(BaseModel):
    prediction_key1: str
    prediction_key2: str
    is_asymmetric: bool = True
    temperature: float = 1.0
    weight: float = 1
    tag: str = "NCELoss"


class NCELoss(Loss):
    def __init__(
        self,
        prediction_key1: str,
        prediction_key2: str,
        is_asymmetric: bool,
        temperature: float,
        weight: float,
        tag: str = "NCELoss",
    ):
        """
        Noise Contrastive Estimation Loss

        Args:
            prediction_key1 (str): key to access embedding 1.
            prediction_key2 (str): key to access embedding 2.
            is_asymmetric (bool, optional): specifies symmetric or asymmetric calculation of NCEloss. Defaults to True.
            temperature (float, optional): temperature. Defaults to 1.0.
            tag (str, optional): Defaults to "NCELoss".
        """
        super().__init__(tag, weight)
        self.prediction_key1 = prediction_key1
        self.prediction_key2 = prediction_key2
        self.is_asymmetric = is_asymmetric
        self.temperature = temperature

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Args:
            forward_batch (InferenceResultBatch): data batch.

        Returns:
            torch.Tensor: loss tensor.
        """
        embedding1 = forward_batch.get_predictions(self.prediction_key1)
        embedding2 = forward_batch.get_predictions(self.prediction_key2)

        contiguous_embedding1 = embedding1.contiguous()
        contiguous_embedding2 = embedding2.contiguous()

        loss = nce_loss(
            contiguous_embedding1, contiguous_embedding2, embedding1.device, self.is_asymmetric, self.temperature
        )
        return loss


class ClipLossConfig(BaseModel):
    logit_scale_key: str
    prediction_key1: str
    prediction_key2: str
    weight: float = 1
    local_loss: bool = True
    tag: str = "ClipLoss"


class ClipLoss(Loss):
    def __init__(
        self,
        logit_scale_key: str,
        prediction_key1: str,
        prediction_key2: str,
        weight: float,
        local_loss: bool,
        tag: str = "ClipLoss",
    ):
        """
        CLIP Loss (Source: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py)

        Args:
            logit_scale_key (str): Value of a learnable logit scale parameter.
            prediction_key1 (str): Key to access embedding 1.
            prediction_key2 (str): Key to access embedding 2.
            tag (str, optional): Defaults to "ClipLoss".
        """
        super().__init__(tag, weight)
        self.logit_scale_key = logit_scale_key
        self.prediction_key1 = prediction_key1
        self.prediction_key2 = prediction_key2
        self.local_loss = local_loss

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Args:
            forward_batch (InferenceResultBatch): data batch.

        Returns:
            torch.Tensor: loss tensor.
        """
        logit_scale = forward_batch.get_predictions(self.logit_scale_key)
        embedding1 = forward_batch.get_predictions(self.prediction_key1).contiguous()
        embedding2 = forward_batch.get_predictions(self.prediction_key2).contiguous()
        device = embedding1.device

        # Gather all embeddings from each rank
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        gathered_embedding1 = [torch.zeros_like(embedding1) for _ in range(world_size)]
        gathered_embedding2 = [torch.zeros_like(embedding2) for _ in range(world_size)]
        dist.all_gather(gathered_embedding1, embedding1)
        dist.all_gather(gathered_embedding2, embedding2)

        # Make sure we have gradients for the "local" embeddings
        if not self.local_loss:
            gathered_embedding1[rank] = embedding1
            gathered_embedding2[rank] = embedding2

        # Combine embeddings
        gathered_embedding1 = torch.cat(gathered_embedding1, dim=0)
        gathered_embedding2 = torch.cat(gathered_embedding2, dim=0)

        # Calculate logits
        if self.local_loss:
            logits_per_embedding1 = logit_scale * embedding1 @ gathered_embedding2.T
            logits_per_embedding2 = logit_scale * embedding2 @ gathered_embedding1.T
        else:
            logits_per_embedding1 = logit_scale * gathered_embedding1 @ gathered_embedding2.T
            logits_per_embedding2 = logits_per_embedding1.T

        # Build gt labels for diagonal
        num_logits = logits_per_embedding1.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        if world_size > 1 and self.local_loss:
            labels = labels + num_logits * rank

        # Calculate loss
        clip_loss = (
            F.cross_entropy(logits_per_embedding1, labels) + F.cross_entropy(logits_per_embedding2, labels)
        ) / 2

        return clip_loss
