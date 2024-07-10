from math import floor
from typing import Annotated, Dict, Optional, Tuple, Union

import torch
from einops.layers.torch import Rearrange
from pydantic import BaseModel, Field
from torch import nn

from modalities.nn.attention import AttentionConfig, AttentionType, MultiHeadAttention
from modalities.nn.mlp import MLP


class VisionTransformerConfig(BaseModel):
    sample_key: str
    prediction_key: str
    img_size: Annotated[Union[Tuple[int, int], int], Field(ge=1)] = 224
    n_classes: Optional[Annotated[int, Field(ge=1)]] = 1000
    n_layer: Annotated[int, Field(ge=1)] = 12
    attention_config: AttentionConfig = None
    n_head: Annotated[int, Field(ge=1)] = 8
    n_embd: Annotated[int, Field(ge=1)] = 768
    ffn_hidden: Annotated[int, Field(ge=1)] = 3072
    dropout: Annotated[float, Field(ge=0.0)] = 0.0
    patch_size: Annotated[int, Field(ge=1)] = 16
    patch_stride: Annotated[int, Field(ge=1)] = 16
    n_img_channels: Annotated[int, Field(ge=1)] = 3
    add_cls_token: bool = True
    bias: bool = True
    num_video_frames: Annotated[int, Field(ge=0)] = 1  # TODO: read this from dataloader/train config
    n_latents: Annotated[int, Field(ge=1)] = 64


class ImagePatchEmbedding(nn.Module):
    def __init__(
        self,
        n_img_channels: int = 3,
        n_embd: int = 768,
        patch_size: int = 16,
        patch_stride: int = 16,
        add_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=n_img_channels, out_channels=n_embd, kernel_size=patch_size, stride=patch_stride
        )

        # Define a rearrangement operation to reshape the tensor from
        # batched 4D format (batch_size, channels, height, width) to
        # batched 3D format (batch_size, height*width, channels).
        # This is required to support torch.compile.
        # See https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
        self.rearrange = Rearrange("b c h w -> b (h w) c")

        self.cls_token = None
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.conv(x)
        x = self.rearrange(x)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.repeat(B, 1, 1), x], dim=1)
        return x


class VideoPatchEmbedding(nn.Module):
    def __init__(
        self,
        n_img_channels: int = 3,
        n_embd: int = 768,
        patch_size: int = 16,
        patch_stride: int = 16,
    ) -> None:
        super().__init__()
        self.input_rearrange = Rearrange("b T c h w -> b c T h w")
        self.conv = nn.Conv3d(
            in_channels=n_img_channels,
            out_channels=n_embd,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_stride),
        )  # TODO: check the 3D conv again

        # See https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
        self.rearrange = Rearrange("b c T h w -> b T (h w) c")  # TODO: this might change when implementing dataloader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_rearrange(x)
        x = self.conv(x)
        x = self.rearrange(x)
        return x  # [b T S D]


class VisionTransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int = 768,
        n_head: int = 8,
        ffn_hidden: int = 3072,
        bias: bool = True,
        dropout: float = 0.0,
        attention_config: AttentionConfig = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attention = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            attention_config=attention_config,
            attention_type=AttentionType.NON_CAUSAL_SELF_ATTENTION,
        )
        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(in_features=n_embd, hidden_features=ffn_hidden, bias=bias, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PerceiverConfig(BaseModel):
    n_embd: Annotated[int, Field(ge=1)] = 768
    n_head: Annotated[int, Field(ge=1)] = 8
    n_layer: Annotated[int, Field(ge=1)] = 12
    n_self_attn_layers_per_block: Annotated[int, Field(ge=1)] = 1  # self attn layers per cross attn layer
    n_latents: Annotated[int, Field(ge=1)] = 64  # seq length of outputs
    n_frames: Annotated[int, Field(ge=1)] = 1  # optional time dimension for inputs
    block_size: Annotated[int, Field(ge=1)] = 196  # seq length of inputs
    ffn_hidden: Annotated[int, Field(ge=1)] = 3072
    bias: bool = True
    dropout: Annotated[float, Field(ge=0.0)] = 0.0
    attention_config: AttentionConfig


class Perceiver(nn.Module):
    """Perceiver / Perceiver Resampler

    This is a transformer based architecture that performs cross and self attention
    to compress and embed sequences of any modality inputs. The purpose is to represent
    potentially long variable-length sequences (such as spatio-temporal embeddings from videos, or audio embeddings)
    by a (shorter) fixed-length sequence.

    The implementation is based on the following paper:
    paper: 'Flamingo: a Visual Language Model for Few-Shot Learning'
    Link: https://github.com/mlfoundations/open_flamingo

    A modification compared to the Perceiver Resampler from Flamingo is that we add self-attention layers
    between the cross-attention layers as in the original Perceiver paper:
    paper: Perceiver: General Perception with Iterative Attention
    Link: http://proceedings.mlr.press/v139/jaegle21a/jaegle21a.pdf


    """

    def __init__(
        self,
        n_embd: int = 768,
        n_head: int = 8,
        n_layer: int = 12,
        n_self_attn_layers_per_block=2,
        n_latents: int = 64,
        n_frames: int = 1,
        block_size: int = 196,
        ffn_hidden: int = 3072,
        bias: bool = True,
        dropout: float = 0.0,
        attention_config: AttentionConfig = None,
    ) -> None:
        super().__init__()
        self.positional_embedding_fn = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embd)
        self.n_frames = n_frames

        if self.n_frames > 1:
            self.time_embd = nn.Parameter(torch.randn(self.n_frames, 1, n_embd))
            self.rearrange = Rearrange("b T S D -> b (T S) D")

        self.perceiver_blocks = nn.ModuleList(
            [
                PerceiverTransformerBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    ffn_hidden=ffn_hidden,
                    bias=bias,
                    dropout=dropout,
                    attention_config=attention_config,
                    n_self_attn_layers=n_self_attn_layers_per_block,
                )
                for _ in range(n_layer)
            ]
        )
        self.latents = nn.Parameter(torch.randn(n_latents, n_embd))  # [R,d]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        if self.n_frames > 1:
            x = self.dropout(x + self.time_embd.repeat(B, 1, 1, 1))
            x = self.dropout(x + self.positional_embedding_fn.weight)
            x = self.rearrange(x)
        else:
            x = self.dropout(x + self.positional_embedding_fn.weight)
        latents = self.latents.repeat(B, 1, 1)  # [b,R,d] with R<<T*S
        for block in self.perceiver_blocks:
            latents = block(x, latents)
        return latents


class PerceiverTransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int = 768,
        n_head: int = 8,
        n_self_attn_layers=2,
        ffn_hidden: int = 3072,
        bias: bool = True,
        dropout: float = 0.0,
        attention_config: AttentionConfig = None,
    ) -> None:
        super().__init__()
        self.norm_latents = nn.LayerNorm(n_embd)
        self.norm_inputs = nn.LayerNorm(n_embd)
        self.cross_attention = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            attention_config=attention_config,
            attention_type=AttentionType.CROSS_ATTENTION,
        )

        self.cross_attn_mlp = nn.Sequential(
            nn.LayerNorm(n_embd), MLP(in_features=n_embd, hidden_features=ffn_hidden, bias=bias, dropout=dropout)
        )
        self.self_attn_layers = nn.ModuleList([])
        for _ in range(n_self_attn_layers):
            self.self_attn_layers.append(
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(n_embd),
                            MultiHeadAttention(
                                n_embd=n_embd,
                                n_head=n_head,
                                attention_config=attention_config,
                                attention_type=AttentionType.NON_CAUSAL_SELF_ATTENTION,
                            ),
                        ),
                        nn.Sequential(
                            nn.LayerNorm(n_embd),
                            MLP(in_features=n_embd, hidden_features=ffn_hidden, bias=bias, dropout=dropout),
                        ),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        latents = self.norm_latents(latents)
        x = self.norm_inputs(x)
        context = torch.cat((x, latents), dim=-2)  # optional?
        latents = latents + self.cross_attention(latents, context=context)
        latents = latents + self.cross_attn_mlp(latents)
        for attn, ff in self.self_attn_layers:
            latents = attn(latents) + latents
            latents = ff(latents) + latents
        return latents


class VisionTransformer(nn.Module):
    """ViT

    The Vision Transformer (ViT) is a pure transformer architecture
    that applies attention mechanisms directly to sequences of image patches for image classification tasks.

    Paper: `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
    Link: https://arxiv.org/abs/2010.11929

    This architecture is extended to encode videos using a perceiver resampler transformer model
    """

    def __init__(
        self,
        sample_key: str,
        prediction_key: str,
        img_size: Union[Tuple[int, int], int] = 224,
        n_classes: int = 1000,
        n_layer: int = 12,
        attention_config: AttentionConfig = None,
        n_head: int = 8,
        n_embd: int = 768,
        ffn_hidden: int = 3072,
        dropout: float = 0.0,
        patch_size: int = 16,
        patch_stride: int = 16,
        n_img_channels: int = 3,
        add_cls_token: bool = True,
        bias: bool = True,
        num_video_frames: int = 1,  # when dealing with video this is bigger than 1
        n_latents: int = 64,
    ) -> None:
        super().__init__()
        self.sample_key = sample_key
        self.has_cls_token = add_cls_token
        self.prediction_key = prediction_key
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.block_size = self._calculate_block_size(self.img_size, patch_size, patch_stride, add_cls_token)
        self.dropout = nn.Dropout(dropout)

        self.head = None
        if n_classes is not None:
            self.norm = nn.LayerNorm(n_embd)
            self.head = nn.Linear(in_features=n_embd, out_features=n_classes, bias=bias)

        self.vision_input = "Image"
        if num_video_frames > 1:  # video data
            self.vision_input = "Video"
            self.embedding_fn = VideoPatchEmbedding(n_img_channels, n_embd, patch_size, patch_stride)  # [b T S D]
            self.time_embd = nn.Parameter(torch.randn(num_video_frames, 1, n_embd))  # [T,1,d]
            if add_cls_token:
                n_latents += 1  # to count for a video level cls token
                self.block_size -= 1
            self.perceiver_resampler = Perceiver(
                n_embd=n_embd,
                n_head=n_head,
                n_layer=n_layer,
                n_latents=n_latents,
                block_size=self.block_size,
                ffn_hidden=ffn_hidden,
                bias=bias,
                dropout=dropout,
                n_self_attn_layers_per_block=1,
                n_frames=num_video_frames,
                attention_config=attention_config,
            )
            self.rearrange = Rearrange("b T S D -> b (T S) D")
        else:
            self.embedding_fn = ImagePatchEmbedding(n_img_channels, n_embd, patch_size, patch_stride, add_cls_token)
            self.positional_embedding_fn = nn.Embedding(num_embeddings=self.block_size, embedding_dim=n_embd)  # [S D]

            self.blocks = nn.ModuleList(
                [
                    VisionTransformerBlock(
                        n_embd=n_embd,
                        n_head=n_head,
                        ffn_hidden=ffn_hidden,
                        bias=bias,
                        dropout=dropout,
                        attention_config=attention_config,
                    )
                    for _ in range(n_layer)
                ]
            )

    def forward_images(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_fn(x)
        x = self.dropout(x + self.positional_embedding_fn.weight)
        for block in self.blocks:
            x = block(x)
        return x

    def forward_videos(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video data into a shorter sequence of tokens

        Args:
            x (torch.Tensor): images from multiple video frames
                shape (b c T h w)
                b: batch size
                T: temporal dim
                h,w: spatial dims (S=h*w)
                c: embedding dim (D)

        Returns:
            torch.Tensor: latents
                shape (b R D) R << T*S
        """
        x = self.embedding_fn(x)  # [b T S D]
        b, T = x.shape[:2]
        latents = self.perceiver_resampler(x)
        return latents

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # TODO video adapt
        # TODO: add video_sample_key and video_prediction_key
        x = inputs[self.sample_key]
        if self.vision_input == "Video":
            x = self.forward_videos(x)
        else:
            x = self.forward_images(x)
        if self.head:
            if self.has_cls_token:
                x = x[:, 0]
            else:
                x = x.mean(dim=1)
            x = self.head(self.norm(x))
        return {self.prediction_key: x}

    @staticmethod
    def _calculate_block_size(img_size: Tuple[int, int], patch_size: int, patch_stride: int, add_cls_token: bool):
        # See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html for details
        block_size = (floor((img_size[0] - patch_size) / patch_stride) + 1) * (
            floor((img_size[1] - patch_size) / patch_stride) + 1
        ) + int(add_cls_token)
        return block_size
