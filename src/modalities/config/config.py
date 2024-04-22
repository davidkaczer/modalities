import os
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Tuple

import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, FilePath, PositiveInt, field_validator, model_validator
from torch.distributed.fsdp import ShardingStrategy
from transformers import GPT2TokenizerFast
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

from modalities.config.lookup_enum import LookupEnum
from modalities.config.pydanctic_if_types import (
    PydanticCheckpointLoadingIFType,
    PydanticCheckpointSavingExecutionIFType,
    PydanticCheckpointSavingStrategyIFType,
    PydanticCollateFnIFType,
    PydanticDatasetIFType,
    PydanticLLMDataLoaderIFType,
    PydanticOptimizerIFType,
    PydanticPytorchDeviceType,
    PydanticPytorchModuleType,
    PydanticSamplerIFType,
    PydanticTokenizerIFType,
)
from modalities.config.utils import parse_torch_device
from modalities.running_env.env_utils import MixedPrecisionSettings, has_bfloat_support
from modalities.util import get_date_of_run, parse_enum_by_name


class ProcessGroupBackendType(LookupEnum):
    nccl = "nccl"


class TokenizerTypes(LookupEnum):
    GPT2TokenizerFast = GPT2TokenizerFast
    LlamaTokenizerFast = LlamaTokenizerFast


class PassType(LookupEnum):
    BY_VALUE = "by_value"
    BY_REFERENCE = "by_reference"


class WandbMode(LookupEnum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    DISABLED = "DISABLED"


class PrecisionEnum(LookupEnum):
    FP32 = torch.float32
    FP16 = torch.float16
    BF16 = torch.bfloat16


class ReferenceConfig(BaseModel):
    instance_key: str
    pass_type: PassType


class CLMCrossEntropyLossConfig(BaseModel):
    target_key: str
    prediction_key: str


# Checkpointing
class SaveEveryKStepsCheckpointingStrategyConfig(BaseModel):
    k: PositiveInt


class SaveKMostRecentCheckpointsStrategyConfig(BaseModel):
    k: Annotated[int, Field(strict=True, ge=-1)]


class TorchCheckpointLoadingConfig(BaseModel):
    device: PydanticPytorchDeviceType
    precision: Optional[PrecisionEnum] = None

    @field_validator("device", mode="before")
    def parse_device(cls, device) -> PydanticPytorchDeviceType:
        return parse_torch_device(device)


class FSDPCheckpointLoadingConfig(BaseModel):
    global_rank: Annotated[int, Field(strict=True, ge=0)]
    block_names: List[str]
    mixed_precision_settings: MixedPrecisionSettings
    sharding_strategy: ShardingStrategy

    @field_validator("mixed_precision_settings", mode="before")
    def parse_mixed_precision_setting_by_name(cls, name):
        mixed_precision_settings: MixedPrecisionSettings = parse_enum_by_name(
            name=name, enum_type=MixedPrecisionSettings
        )
        if not has_bfloat_support() and (
            mixed_precision_settings == MixedPrecisionSettings.BF_16
            or mixed_precision_settings == MixedPrecisionSettings.BF_16_WORKING
        ):
            raise ValueError("BF16 not supported in the current environment")
        return mixed_precision_settings

    @field_validator("sharding_strategy", mode="before")
    def parse_sharding_strategy_by_name(cls, name):
        return parse_enum_by_name(name=name, enum_type=ShardingStrategy)


class FSDPCheckpointSavingConfig(BaseModel):
    checkpoint_path: Path
    global_rank: Annotated[int, Field(strict=True, ge=0)]
    experiment_id: str


class CheckpointSavingConfig(BaseModel):
    checkpoint_saving_strategy: PydanticCheckpointSavingStrategyIFType
    checkpoint_saving_execution: PydanticCheckpointSavingExecutionIFType


class AdamOptimizerConfig(BaseModel):
    lr: float
    wrapped_model: PydanticPytorchModuleType
    betas: Tuple[float, float]
    eps: float
    weight_decay: float


class AdamWOptimizerConfig(BaseModel):
    lr: float
    wrapped_model: PydanticPytorchModuleType
    betas: Tuple[float, float]
    eps: float
    weight_decay: float


class DummyLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType


class StepLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType
    step_size: Annotated[int, Field(strict=True, gt=0)]
    gamma: Annotated[float, Field(strict=True, ge=0.0)]
    last_epoch: Annotated[int, Field(strict=True, ge=-1)] = -1
    verbose: bool = False


class OneCycleLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType
    max_lr: Annotated[float, Field(strict=True, gt=0.0)] | List[Annotated[float, Field(strict=True, gt=0.0)]]
    total_steps: Optional[Annotated[int, Field(strict=True, gt=0)]] = None
    epochs: Optional[Annotated[int, Field(strict=True, gt=0)]] = None
    steps_per_epoch: Optional[Annotated[int, Field(strict=True, gt=0)]] = None
    pct_start: Annotated[float, Field(strict=True, gt=0.0, le=1.0)]
    anneal_strategy: str
    cycle_momentum: bool = True
    base_momentum: Annotated[float, Field(strict=True, gt=0)] | List[
        Annotated[float, Field(strict=True, gt=0.0)]
    ] = 0.85
    max_momentum: Annotated[float, Field(strict=True, gt=0.0)] | List[
        Annotated[float, Field(strict=True, gt=0.0)]
    ] = 0.95
    div_factor: Annotated[float, Field(strict=True, gt=0.0)]
    final_div_factor: Annotated[float, Field(strict=True, gt=0.0)]
    three_phase: bool = False
    last_epoch: Annotated[int, Field(strict=True, ge=-1)] = -1
    verbose: bool = False

    @model_validator(mode="after")
    def check_totals_steps_and_epchs(self) -> "OneCycleLRSchedulerConfig":
        if self.total_steps is None and (self.epochs is None or self.steps_per_epoch is None):
            raise ValueError("Please define total_steps or (epochs and steps_per_epoch).")
        return self


class ConstantLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType
    factor: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    total_iters: Annotated[int, Field(strict=True, gt=0)]
    last_epoch: Annotated[int, Field(strict=True, ge=-1)] = -1
    verbose: bool = False


class CosineAnnealingLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType
    t_max: Annotated[int, Field(strict=True, gt=0)]
    eta_min: Annotated[float, Field(strict=True, ge=0.0)]
    last_epoch: Annotated[int, Field(strict=True, ge=-1)] = -1
    verbose: bool = False


class CheckpointedOptimizerConfig(BaseModel):
    checkpoint_loading: PydanticCheckpointLoadingIFType
    checkpoint_path: Path
    wrapped_model: PydanticPytorchModuleType
    optimizer: PydanticOptimizerIFType


class CheckpointedModelConfig(BaseModel):
    checkpoint_loading: PydanticCheckpointLoadingIFType
    checkpoint_path: Path
    model: PydanticPytorchModuleType


class FSDPWrappedModelConfig(BaseModel):
    model: PydanticPytorchModuleType
    sync_module_states: bool
    mixed_precision_settings: MixedPrecisionSettings
    sharding_strategy: ShardingStrategy
    block_names: List[str]

    @field_validator("mixed_precision_settings", mode="before")
    def parse_mixed_precision_setting_by_name(cls, name):
        mixed_precision_settings: MixedPrecisionSettings = parse_enum_by_name(
            name=name, enum_type=MixedPrecisionSettings
        )
        if not has_bfloat_support() and (
            mixed_precision_settings == MixedPrecisionSettings.BF_16
            or mixed_precision_settings == MixedPrecisionSettings.BF_16_WORKING
        ):
            raise ValueError("BF16 not supported in the current environment")
        return mixed_precision_settings

    @field_validator("sharding_strategy", mode="before")
    def parse_sharding_strategy_by_name(cls, name):
        return parse_enum_by_name(name=name, enum_type=ShardingStrategy)


class PreTrainedHFTokenizerConfig(BaseModel):
    pretrained_model_name_or_path: str
    max_length: Annotated[int, Field(strict=True, ge=0)]
    truncation: bool = False
    padding: bool | str = False


class PreTrainedSPTokenizerConfig(BaseModel):
    tokenizer_model_file: str


class DistributedSamplerConfig(BaseModel):
    rank: Annotated[int, Field(strict=True, ge=0)]
    num_replicas: Annotated[int, Field(strict=True, ge=0)]
    shuffle: bool
    dataset: PydanticDatasetIFType
    seed: Optional[int] = 0


class MemMapDatasetConfig(BaseModel):
    raw_data_path: FilePath
    index_path: Optional[FilePath] = None
    block_size: Annotated[int, Field(strict=True, gt=0)]
    tokenizer: PydanticTokenizerIFType
    jq_pattern: str
    sample_key: str


class PackedMemMapDatasetContinuousConfig(BaseModel):
    raw_data_path: Path
    block_size: Annotated[int, Field(strict=True, gt=0)]
    sample_key: str


class PackedMemMapDatasetMegatronConfig(BaseModel):
    raw_data_path: Path
    block_size: Annotated[int, Field(strict=True, gt=0)]
    sample_key: str


class MMapIndexedDatasetConfig(BaseModel):
    path: Path
    skip_warmup: bool


class OpenGPTXMMapDatasetConfig(BaseModel):
    num_samples: Annotated[int, Field(strict=True, ge=1)]
    path: FilePath
    sample_key: str
    sequence_len: PositiveInt


class BatchSamplerConfig(BaseModel):
    sampler: PydanticSamplerIFType
    batch_size: Annotated[int, Field(strict=True, gt=0)]
    drop_last: Literal[True] = True


class ResumableBatchSamplerConfig(BaseModel):
    sampler: PydanticSamplerIFType
    start_index: Annotated[int, Field(strict=True, gt=0)]


class GPT2LLMCollateFnConfig(BaseModel):
    sample_key: str
    target_key: str


class LLMDataLoaderConfig(BaseModel):
    dataloader_tag: str
    dataset: PydanticDatasetIFType
    batch_sampler: PydanticSamplerIFType
    collate_fn: Optional[PydanticCollateFnIFType] = None
    num_workers: Annotated[int, Field(strict=True, ge=0)]
    pin_memory: bool
    shuffle: bool
    skip_num_steps: Optional[int] = 0


class RepeatingDataLoaderConfig(BaseModel):
    dataloader: PydanticLLMDataLoaderIFType
    reshuffle_after_epoch: Optional[bool] = False
    num_epochs: Annotated[int, Field(strict=True, ge=1)]


class DummyProgressSubscriberConfig(BaseModel):
    pass


class SimpleProgressSubscriberConfig(BaseModel):
    train_dataloader: PydanticLLMDataLoaderIFType
    eval_dataloaders: Optional[List[PydanticLLMDataLoaderIFType]] = Field(default_factory=list)
    world_size: int
    global_num_seen_samples: int
    local_rank: int


class RichProgressSubscriberConfig(BaseModel):
    train_dataloader: PydanticLLMDataLoaderIFType
    eval_dataloaders: Optional[List[PydanticLLMDataLoaderIFType]] = Field(default_factory=list)
    global_num_seen_steps: int
    local_rank: int


class DummyResultSubscriberConfig(BaseModel):
    pass


class WandBEvaluationResultSubscriberConfig(BaseModel):
    local_rank: int
    project: str
    experiment_id: str
    mode: WandbMode
    directory: Path
    config_file_path: Path


class RichResultSubscriberConfig(BaseModel):
    num_ranks: int
    local_rank: int


def load_app_config_dict(config_file_path: Path) -> Dict:
    def cuda_env_resolver_fun(var_name: str) -> int:
        int_env_variable_names = ["LOCAL_RANK", "WORLD_SIZE", "RANK"]
        return int(os.getenv(var_name)) if var_name in int_env_variable_names else os.getenv(var_name)

    def modalities_env_resolver_fun(var_name: str) -> int:
        if var_name == "experiment_id":
            return get_date_of_run()
        if var_name == "config_file_path":
            return config_file_path

    def node_env_resolver_fun(var_name: str) -> int:
        if var_name == "num_cpus":
            return os.cpu_count()

    OmegaConf.register_new_resolver("cuda_env", cuda_env_resolver_fun, replace=True)
    OmegaConf.register_new_resolver("modalities_env", modalities_env_resolver_fun, replace=True)
    OmegaConf.register_new_resolver("node_env", node_env_resolver_fun, replace=True)

    cfg = OmegaConf.load(config_file_path)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    return config_dict
