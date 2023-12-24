from typing import Any, Dict, List

import torch.optim as optim
from class_resolver import ClassResolver
from pydantic import BaseModel
from torch.utils.data import DataLoader, Sampler
from transformers import PreTrainedTokenizer

from llm_gym.config.config import AppConfig, OptimizerTypes, SchedulerTypes
from llm_gym.config.lookup_types import (
    CollatorTypes,
    DataloaderTypes,
    DatasetTypes,
    LossTypes,
    ModelTypes,
    SamplerTypes,
    TokenizerTypes,
)
from llm_gym.dataloader.dataset import Dataset
from llm_gym.fsdp.fsdp_running_env import FSDPRunningEnv, RunningEnv, RunningEnvTypes
from llm_gym.loss_functions import CLMCrossEntropyLoss, Loss
from llm_gym.models.gpt2.collator import GPT2LLMCollator
from llm_gym.models.gpt2.gpt2_model import GPT2LLM, NNModel


class ResolverRegister:
    def __init__(self, config: AppConfig) -> None:
        self._resolver_register: Dict[str, ClassResolver] = self._create_resolver_register(config=config)

    def build_component_by_config(self, config: BaseModel, extra_kwargs: Dict = {}) -> Any:
        assert (
            "type_hint" in config.model_fields.keys()
        ), f"Field 'type_hint' missing but needed for initalisation in {config}"

        kwargs = {key: getattr(config.config, key) for key in config.config.model_dump().keys()}
        kwargs.update(extra_kwargs)  # allow override via extra_kwargs, to add nested objects
        return self._build_component(
            register_key=config.type_hint,
            register_query=config.type_hint.name,
            extra_kwargs=kwargs,
        )

    def build_component_by_key_query(self, register_key: str, type_hint: str, extra_kwargs: Dict = {}) -> Any:
        return self._build_component(register_key=register_key, register_query=type_hint, extra_kwargs=extra_kwargs)

    def _build_component(self, register_key: str, register_query: str, extra_kwargs: Dict = {}):
        return self._resolver_register[register_key].make(
            query=register_query,
            pos_kwargs=extra_kwargs,
        )

    def _find_values_with_key_in_nested_structure(self, nested_structure: Dict, key: str) -> List[Any]:
        found_values = []
        for k, v in nested_structure.items():
            if k == key:
                found_values.append(v)
            elif isinstance(v, dict):
                found_values.extend(self._find_values_with_key_in_nested_structure(v, key))
        return found_values

    def _create_resolver_register(self, config: AppConfig) -> Dict[str, ClassResolver]:
        expected_resolvers = set(
            self._find_values_with_key_in_nested_structure(nested_structure=config.model_dump(), key="type_hint")
        )
        resolvers = {
            config.running_env.type_hint: ClassResolver(
                [t.value for t in RunningEnvTypes],
                base=RunningEnv,
                default=FSDPRunningEnv,
            ),
            config.model.type_hint: ClassResolver(
                [t.value for t in ModelTypes],
                base=NNModel,
                default=GPT2LLM,
            ),
            config.optimizer.type_hint: ClassResolver(
                [t.value for t in OptimizerTypes],
                base=optim.Optimizer,
                default=optim.AdamW,
            ),
            config.scheduler.type_hint: ClassResolver(
                [t.value for t in SchedulerTypes],
                base=optim.lr_scheduler.LRScheduler,
                default=optim.lr_scheduler.StepLR,
            ),
            config.loss.type_hint: ClassResolver(
                [t.value for t in LossTypes],
                base=Loss,
                default=CLMCrossEntropyLoss,
            ),
            config.training.train_dataloader.config.dataset.config.tokenizer.type_hint: ClassResolver(
                [t.value for t in TokenizerTypes],
                base=PreTrainedTokenizer,
            ),
            config.training.train_dataloader.config.dataset.type_hint: ClassResolver(
                [t.value for t in DatasetTypes],
                base=Dataset,
            ),
            config.training.train_dataloader.config.sampler.type_hint: ClassResolver(
                [t.value for t in SamplerTypes],
                base=Sampler,
            ),
            config.training.train_dataloader.type_hint: ClassResolver(
                [t.value for t in DataloaderTypes],
                base=DataLoader,
            ),
            config.training.train_dataloader.config.collate_fn.type_hint: ClassResolver(
                [t.value for t in CollatorTypes],
                base=GPT2LLMCollator,
            ),
        }
        assert set(expected_resolvers) == set(
            resolvers
        ), f"Some resolvers are not registered: {set(expected_resolvers).symmetric_difference(resolvers)}"
        return resolvers

    def add_resolver(self, resolver_key: str, resolver: ClassResolver):
        self._resolver_register[resolver_key] = resolver