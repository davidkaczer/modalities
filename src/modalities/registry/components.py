from dataclasses import dataclass
from typing import Callable, Type

import torch
import torch.nn as nn
import transformers
from pydantic import BaseModel
from torch.utils.data import BatchSampler, DistributedSampler

from modalities.checkpointing.checkpoint_saving import CheckpointSaving
from modalities.checkpointing.checkpoint_saving_strategies import (
    SaveEveryKStepsCheckpointingStrategy,
    SaveKMostRecentCheckpointsStrategy,
)
from modalities.checkpointing.fsdp.fsdp_checkpoint_loading import FSDPCheckpointLoading
from modalities.checkpointing.fsdp.fsdp_checkpoint_saving import FSDPCheckpointSaving
from modalities.checkpointing.torch.torch_checkpoint_loading import TorchCheckpointLoading
from modalities.checkpointing.torch.torch_checkpoint_saving import TorchCheckpointSaving
from modalities.config.config import (
    AdamOptimizerConfig,
    AdamWOptimizerConfig,
    ArrowDatasetAudioConfig,
    ArrowDatasetAVConfig,
    ArrowDatasetVisionConfig,
    BatchSamplerConfig,
    CheckpointedModelConfig,
    CheckpointedOptimizerConfig,
    CheckpointSavingConfig,
    ConstantLRSchedulerConfig,
    CosineAnnealingLRSchedulerConfig,
    CosineAnnealingWithWarmupLRSchedulerConfig,
    DistributedSamplerConfig,
    DummyLRSchedulerConfig,
    DummyProgressSubscriberConfig,
    DummyResultSubscriberConfig,
    FSDPCheckpointLoadingConfig,
    FSDPCheckpointSavingConfig,
    FSDPWrappedModelConfig,
    GPT2LLMCollateFnConfig,
    LLMDataLoaderConfig,
    MemMapDatasetConfig,
    OneCycleLRSchedulerConfig,
    OpenGPTXMMapDatasetConfig,
    PackedMemMapDatasetContinuousConfig,
    PackedMemMapDatasetMegatronConfig,
    PreTrainedHFTokenizerConfig,
    PreTrainedSPTokenizerConfig,
    RepeatingDataLoaderConfig,
    RichProgressSubscriberConfig,
    RichResultSubscriberConfig,
    SaveEveryKStepsCheckpointingStrategyConfig,
    SaveKMostRecentCheckpointsStrategyConfig,
    SimpleProgressSubscriberConfig,
    StepLRSchedulerConfig,
    TorchCheckpointLoadingConfig,
    TorchCheckpointSavingConfig,
    TorchModelConfig,
    WandBEvaluationResultSubscriberConfig,
    WebLoaderConfig,
)
from modalities.dataloader.dataloader_factory import DataloaderFactory
from modalities.dataloader.dataset import (
    AudioTransform,
    AudioTransformConfig,
    DummyDatasetConfig,
    ImageTransform,
    ImageTransformConfig,
    MultimodalWebDataset,
    MultimodalWebDatasetBuilder,
    MultimodalWebDatasetBuilderConfig,
    MultimodalWebDatasetConfig,
    TextTransform,
    TextTransformConfig,
    VideoTransform,
    VideoTransformConfig,
)
from modalities.dataloader.dataset_factory import DatasetFactory
from modalities.logging_broker.subscriber_impl.subscriber_factory import (
    ProgressSubscriberFactory,
    ResultsSubscriberFactory,
)
from modalities.loss_functions import (
    ClipLoss,
    ClipLossConfig,
    CrossEntropyLoss,
    CrossEntropyLossConfig,
    NCELoss,
    NCELossConfig,
)
from modalities.models.anymal.anymal_model import AnyMAL, AnyMALConfig
from modalities.models.anymal.collator import AnyMALCollateFnConfig, AnyMALCollatorFn
from modalities.models.audio_transformer.audio_transformer_model import AudioTransformer, AudioTransformerConfig
from modalities.models.coca.coca_model import CoCa, CoCaConfig
from modalities.models.coca.collator import CoCaCollateFnConfig, CoCaCollatorFn
from modalities.models.components.layer_norms import LayerNormConfig, RMSLayerNorm, RMSLayerNormConfig
from modalities.models.gpt2.collator import GPT2LLMCollateFn
from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2LLMConfig
from modalities.models.huggingface.huggingface_models import (
    HuggingFacePretrainedModel,
    HuggingFacePretrainedModelConfig,
)

# from modalities.models.mamba.mamba_config import MambaLLMConfig
# from modalities.models.mamba.mamba_model import MambaLLM
from modalities.models.model_factory import ModelFactory
from modalities.models.vision_transformer.vision_transformer_model import VisionTransformer, VisionTransformerConfig
from modalities.optimizers.lr_schedulers import DummyLRScheduler
from modalities.optimizers.optimizer_factory import OptimizerFactory
from modalities.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer, PreTrainedSPTokenizer
from modalities.training.gradient_clipping.fsdp.fsdp_gradient_clipper import (
    DummyGradientClipper,
    FSDPGradientClipper,
    FSDPLoggingOnlyGradientClipper,
)
from modalities.training.gradient_clipping.fsdp.fsdp_gradient_clipper_config import (
    DummyGradientClipperConfig,
    FSDPDummyGradientClipperConfig,
    FSDPGradientClipperConfig,
)
from modalities.training.gradient_clipping.torch.torch_gradient_clipper import TorchGradientClipper
from modalities.training.gradient_clipping.torch.torch_gradient_clipper_config import TorchGradientClipperConfig


@dataclass
class ComponentEntity:
    component_key: str
    variant_key: str
    component_type: Type | Callable
    component_config_type: Type[BaseModel]


COMPONENTS = [
    # models
    ComponentEntity("model", "gpt2", GPT2LLM, GPT2LLMConfig),
    # ComponentEntity("model", "mamba", MambaLLM, MambaLLMConfig),
    ComponentEntity(
        "model", "huggingface_pretrained_model", HuggingFacePretrainedModel, HuggingFacePretrainedModelConfig
    ),
    ComponentEntity("model", "checkpointed", ModelFactory.get_checkpointed_model, CheckpointedModelConfig),
    ComponentEntity("model", "fsdp_wrapped", ModelFactory.get_fsdp_wrapped_model, FSDPWrappedModelConfig),
    ComponentEntity("model", "torch_model", ModelFactory.get_torch_model, TorchModelConfig),
    ComponentEntity("model", "coca", CoCa, CoCaConfig),
    ComponentEntity("model", "anymal", AnyMAL, AnyMALConfig),
    ComponentEntity("model", "vit", VisionTransformer, VisionTransformerConfig),
    ComponentEntity("model", "aut", AudioTransformer, AudioTransformerConfig),
    # losses
    ComponentEntity("loss", "cross_entropy_loss", CrossEntropyLoss, CrossEntropyLossConfig),
    ComponentEntity("loss", "nce_loss", NCELoss, NCELossConfig),
    ComponentEntity("loss", "clip_loss", ClipLoss, ClipLossConfig),
    # optmizers
    ComponentEntity("optimizer", "adam", OptimizerFactory.get_adam, AdamOptimizerConfig),
    ComponentEntity("optimizer", "adam_w", OptimizerFactory.get_adam_w, AdamWOptimizerConfig),
    ComponentEntity(
        "optimizer", "checkpointed", OptimizerFactory.get_checkpointed_optimizer, CheckpointedOptimizerConfig
    ),
    # schedulers
    ComponentEntity("scheduler", "dummy_lr", DummyLRScheduler, DummyLRSchedulerConfig),
    ComponentEntity("scheduler", "step_lr", torch.optim.lr_scheduler.StepLR, StepLRSchedulerConfig),
    ComponentEntity("scheduler", "constant_lr", torch.optim.lr_scheduler.ConstantLR, ConstantLRSchedulerConfig),
    ComponentEntity("scheduler", "onecycle_lr", torch.optim.lr_scheduler.OneCycleLR, OneCycleLRSchedulerConfig),
    ComponentEntity(
        "scheduler", "cosine_annealing_lr", torch.optim.lr_scheduler.CosineAnnealingLR, CosineAnnealingLRSchedulerConfig
    ),
    ComponentEntity(
        "scheduler",
        "cosine_annealing_with_warmup_lr",
        transformers.get_linear_schedule_with_warmup,
        CosineAnnealingWithWarmupLRSchedulerConfig,
    ),
    # tokenizers
    ComponentEntity("tokenizer", "pretrained_hf_tokenizer", PreTrainedHFTokenizer, PreTrainedHFTokenizerConfig),
    ComponentEntity("tokenizer", "pretrained_sp_tokenizer", PreTrainedSPTokenizer, PreTrainedSPTokenizerConfig),
    # ComponentEntity("tokenizer", "llama_tokenizer_fast", GPT2TokenizerFast, None),  # TODO
    # datasets
    ComponentEntity("dataset", "mem_map_dataset", DatasetFactory.get_mem_map_dataset, MemMapDatasetConfig),
    ComponentEntity(
        "dataset",
        "packed_mem_map_dataset_continuous",
        DatasetFactory.get_packed_mem_map_dataset_continuous,
        PackedMemMapDatasetContinuousConfig,
    ),
    ComponentEntity(
        "dataset",
        "packed_mem_map_dataset_megatron",
        DatasetFactory.get_packed_mem_map_dataset_megatron,
        PackedMemMapDatasetMegatronConfig,
    ),
    ComponentEntity(
        "dataset", "open_gptx_mmap_dataset", DatasetFactory.get_open_gptx_mmap_dataset, OpenGPTXMMapDatasetConfig
    ),
    ComponentEntity("dataset", "dummy_dataset", DatasetFactory.get_dummy_dataset, DummyDatasetConfig),
    ComponentEntity("dataset", "web_dataset", MultimodalWebDataset, MultimodalWebDatasetConfig),
    ComponentEntity("dataset", "web_dataset_builder", MultimodalWebDatasetBuilder, MultimodalWebDatasetBuilderConfig),
    # Data transforms & augmentations
    ComponentEntity("transform", "text_transform", TextTransform, TextTransformConfig),
    ComponentEntity("transform", "image_transform", ImageTransform, ImageTransformConfig),
    ComponentEntity("transform", "audio_transform", AudioTransform, AudioTransformConfig),
    ComponentEntity("transform", "video_transform", VideoTransform, VideoTransformConfig),
    ComponentEntity(
        "dataset", "arrow_dataset_vision", DatasetFactory.get_arrow_dataset_vision, ArrowDatasetVisionConfig
    ),
    ComponentEntity("dataset", "arrow_dataset_audio", DatasetFactory.get_arrow_dataset_audio, ArrowDatasetAudioConfig),
    ComponentEntity("dataset", "arrow_dataset_av", DatasetFactory.get_arrow_dataset_av, ArrowDatasetAVConfig),
    # samplers
    ComponentEntity("sampler", "distributed_sampler", DistributedSampler, DistributedSamplerConfig),
    # batch samplers
    ComponentEntity("batch_sampler", "default", BatchSampler, BatchSamplerConfig),
    # collators
    ComponentEntity("collate_fn", "gpt_2_llm_collator", GPT2LLMCollateFn, GPT2LLMCollateFnConfig),
    ComponentEntity("collate_fn", "coca_collator", CoCaCollatorFn, CoCaCollateFnConfig),
    ComponentEntity("collate_fn", "anymal_collator", AnyMALCollatorFn, AnyMALCollateFnConfig),
    # data loaders
    ComponentEntity("data_loader", "default", DataloaderFactory.get_dataloader, LLMDataLoaderConfig),
    ComponentEntity("data_loader", "web_loader", DataloaderFactory.get_web_loader, WebLoaderConfig),
    ComponentEntity(
        "data_loader", "repeating_data_loader", DataloaderFactory.get_repeating_dataloader, RepeatingDataLoaderConfig
    ),
    # checkpointing
    ComponentEntity("checkpoint_saving", "default", CheckpointSaving, CheckpointSavingConfig),
    # checkpointing strategies
    ComponentEntity(
        "checkpoint_saving_strategy",
        "save_every_k_steps_checkpointing_strategy",
        SaveEveryKStepsCheckpointingStrategy,
        SaveEveryKStepsCheckpointingStrategyConfig,
    ),
    ComponentEntity(
        "checkpoint_saving_strategy",
        "save_k_most_recent_checkpoints_strategy",
        SaveKMostRecentCheckpointsStrategy,
        SaveKMostRecentCheckpointsStrategyConfig,
    ),
    # checkpoint saving execution
    ComponentEntity("checkpoint_saving_execution", "fsdp", FSDPCheckpointSaving, FSDPCheckpointSavingConfig),
    ComponentEntity("checkpoint_saving_execution", "torch", TorchCheckpointSaving, TorchCheckpointSavingConfig),
    # checkpoint loading
    ComponentEntity("checkpoint_loading", "fsdp", FSDPCheckpointLoading, FSDPCheckpointLoadingConfig),
    ComponentEntity("checkpoint_loading", "torch", TorchCheckpointLoading, TorchCheckpointLoadingConfig),
    # Progress subscriber
    ComponentEntity(
        "progress_subscriber",
        "dummy",
        ProgressSubscriberFactory.get_dummy_progress_subscriber,
        DummyProgressSubscriberConfig,
    ),
    ComponentEntity(
        "progress_subscriber",
        "simple",
        ProgressSubscriberFactory.get_simple_progress_subscriber,
        SimpleProgressSubscriberConfig,
    ),
    ComponentEntity(
        "progress_subscriber",
        "rich",
        ProgressSubscriberFactory.get_rich_progress_subscriber,
        RichProgressSubscriberConfig,
    ),
    # Results subscriber
    ComponentEntity(
        "results_subscriber", "dummy", ResultsSubscriberFactory.get_dummy_result_subscriber, DummyResultSubscriberConfig
    ),
    ComponentEntity(
        "results_subscriber", "rich", ResultsSubscriberFactory.get_rich_result_subscriber, RichResultSubscriberConfig
    ),
    ComponentEntity(
        "results_subscriber",
        "wandb",
        ResultsSubscriberFactory.get_wandb_result_subscriber,
        WandBEvaluationResultSubscriberConfig,
    ),
    # layer norms
    ComponentEntity("layer_norm", "rms_norm", RMSLayerNorm, RMSLayerNormConfig),
    ComponentEntity("layer_norm", "layer_norm", nn.LayerNorm, LayerNormConfig),
    # gradient clippers
    ComponentEntity("gradient_clipper", "fsdp", FSDPGradientClipper, FSDPGradientClipperConfig),
    ComponentEntity(
        "gradient_clipper", "fsdp_logging_only", FSDPLoggingOnlyGradientClipper, FSDPDummyGradientClipperConfig
    ),
    ComponentEntity("gradient_clipper", "dummy", DummyGradientClipper, DummyGradientClipperConfig),
    ComponentEntity("gradient_clipper", "torch", TorchGradientClipper, TorchGradientClipperConfig),
]
