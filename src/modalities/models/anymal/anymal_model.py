from typing import Dict

import torch
from pydantic import BaseModel
from torch import nn

from modalities.config.lookup_enum import LookupEnum
from modalities.models.audio_transformer.audio_transformer_model import AudioTransformer, AudioTransformerConfig
from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2LLMConfig
from modalities.models.model import NNModel
from modalities.models.vision_transformer.vision_transformer_model import (
    PerceiverResampler,
    PerceiverResamplerConfig,
    VisionTransformer,
    VisionTransformerConfig,
)


class AnyMALTrainingStage(LookupEnum):
    MODALITY_ALIGNMENT = "modality_alignment"
    INSTRUCTION_TUNING = "instruction_tuning"


class AnyMALConfig(BaseModel):
    """

    Args:
        text_decoder_config (`GPT2LLMConfig`):
            config for the already trained LLM
        prediction_key (`str`):
            key for the output logits of the model
        text_decoder_checkpoint_path (`str`):
            path to trained checkpoint for text decoder
        vision_encoder_config (`VisionTransformerConfig`), required if training an image-text model:
            config for trained vision encoder
        vision_projector_config (`PerceiverResamplerConfig`), required if training an image-text model:
            config for Perceiver Resampler, which projects embeddings from the vision encoder
            to the text token embedding space
        vision_encoder_checkpoint_path (`str`), required if training an image-text model:
            path to trained checkpoint for image encoder
        audio_encoder_config (`AudioTransformerConfig`), required if training an audio-text model:
            config for trained audio encoder
        audio_projector_config (`PerceiverResamplerConfig`), required if training an audio-text model:
            config for Perceiver Resampler, which projects embeddings from the audio encoder
            to the text token embedding space
        audio_encoder_checkpoint_path (`str`), required if training an audio-text model:
            path to trained checkpoint for audio encoder
        training_stage (`AnyMALTrainingStage`):
            training stage for the model. The first modality alignment stage trains the modality projector using
            captioned images/audio. The second instruction fine-tuning stage trains the modality projector
            and fine-tunes the LLM (using LoRA) using instructions, request and response.


    """

    text_decoder_config: GPT2LLMConfig
    prediction_key: str
    text_decoder_checkpoint_path: str = None
    vision_encoder_config: VisionTransformerConfig = None
    vision_projector_config: PerceiverResamplerConfig = None
    vision_encoder_checkpoint_path: str = None
    audio_encoder_config: AudioTransformerConfig = None
    audio_projector_config: PerceiverResamplerConfig = None
    audio_encoder_checkpoint_path: str = None
    training_stage: AnyMALTrainingStage = AnyMALTrainingStage.MODALITY_ALIGNMENT


class AnyMAL(NNModel):
    """Implementation of the AnyMAL model for multimodal alignment and instruction tuning of LLMs.
    Based on the paper:AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model
    Link: https://arxiv.org/abs/2309.16058
    """

    def __init__(
        self,
        text_decoder_config: GPT2LLMConfig,
        prediction_key: str,
        text_decoder_checkpoint_path: str = None,
        vision_encoder_config: VisionTransformerConfig = None,
        vision_projector_config: PerceiverResamplerConfig = None,
        vision_encoder_checkpoint_path: str = None,
        audio_encoder_config: AudioTransformerConfig = None,
        audio_projector_config: PerceiverResamplerConfig = None,
        audio_encoder_checkpoint_path: str = None,
        training_stage: AnyMALTrainingStage = AnyMALTrainingStage.MODALITY_ALIGNMENT,
        seed: int = None,
    ) -> None:
        super().__init__(seed=seed)
        self.prediction_key = prediction_key
        self.vision_encoder_config = vision_encoder_config
        self.audio_encoder_config = audio_encoder_config
        self.text_decoder_config = text_decoder_config
        self.training_stage = training_stage

        if (vision_encoder_config is not None and audio_encoder_config is not None) or (
            vision_encoder_config is None and audio_encoder_config is None
        ):
            raise ValueError("Either a vision or audio encoder should be specified.")

        if vision_encoder_config is not None:
            if vision_projector_config is None:
                raise ValueError("Vision projector should not be None.")
            self.modality_prediction_key = vision_encoder_config.prediction_key
            self.modality_encoder = VisionTransformer(**dict(vision_encoder_config))
            vision_projector_config.block_size = self.modality_encoder.block_size
            self.modality_projector = PerceiverResampler(**dict(vision_projector_config))
            if vision_encoder_checkpoint_path is not None:
                checkpoint = torch.load(vision_encoder_checkpoint_path)
                self.modality_encoder.load_state_dict(checkpoint)
        elif audio_encoder_config is not None:
            if audio_projector_config is None:
                raise ValueError("Audio projector should not be None.")
            self.modality_prediction_key = audio_encoder_config.prediction_key
            self.modality_encoder = AudioTransformer(**dict(audio_encoder_config))
            audio_projector_config.block_size = self.modality_encoder.block_size
            self.modality_projector = nn.Sequential(
                nn.Linear(self.audio_encoder_config.n_embd, audio_projector_config.n_embd),
                PerceiverResampler(**dict(audio_projector_config)),
            )
            if audio_encoder_checkpoint_path is not None:
                checkpoint = torch.load(audio_encoder_checkpoint_path)
                self.modality_encoder.load_state_dict(checkpoint)

        self.text_decoder = GPT2LLM(**dict(text_decoder_config))
        if text_decoder_checkpoint_path is not None:
            checkpoint = torch.load(text_decoder_checkpoint_path)
            for key in list(checkpoint.keys()):
                if "lm_head_norm.weight" in key:
                    checkpoint[key.replace("lm_head_norm.weight", "ln_f.gain")] = checkpoint.pop(key)
                elif "lm_head_norm.bias" in key:
                    checkpoint[key.replace("lm_head_norm.bias", "ln_f.bias_tensor")] = checkpoint.pop(key)
                elif "norm.weight" in key:
                    checkpoint[key.replace("norm.weight", "norm.gain")] = checkpoint.pop(key)
                elif "norm.bias" in key:
                    checkpoint[key.replace("norm.bias", "norm.bias_tensor")] = checkpoint.pop(key)

            self.text_decoder.load_state_dict(checkpoint)

        """
        # Freeze parameters and/or initialize LoRA based on training stage
        # currently does not work with FSDP

        for param in self.modality_encoder.parameters():
            param.requires_grad = False

        if training_stage == AnyMALTrainingStage.MODALITY_ALIGNMENT:
            for param in self.text_decoder.parameters():
                param.requires_grad = False
        elif training_stage == AnyMALTrainingStage.INSTRUCTION_TUNING:
            peft_config = LoraConfig(
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_attn", "k_attn", "v_attn", "c_proj", "c_fc"] # TODO: make these configurable
            )
            self.text_decoder = get_peft_model(self.text_decoder, peft_config)
        """

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        modality_emb = self.modality_encoder(inputs)[self.modality_prediction_key]
        proj_modality_emb = self.modality_projector(modality_emb)
        text_emb = self.text_decoder.get_input_embeddings()(inputs[self.text_decoder_config.sample_key])
        # prepend projected modality embeddings to token embeddings
        input_emb = torch.cat((proj_modality_emb, text_emb), axis=1)
        inputs["input_emb"] = input_emb

        text_logits = self.text_decoder(inputs)[self.text_decoder_config.prediction_key]
        return {self.prediction_key: text_logits}
