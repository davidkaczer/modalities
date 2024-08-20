from enum import Enum
from typing import Dict

from modalities.config.instantiation_models import WebAgentLLMInstantiationModel
from pydantic import BaseModel

from modalities.config.component_factory import ComponentFactory
from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.webagent_llm_part.config import WebAgentLLMConfig
from modalities.webagent_llm_part.webagent_llm_component import WebAgentLLMComponent


def get_concatenated_model(config: Dict):
    registry = Registry(COMPONENTS)
    registry.add_entity(
        component_key="webAgent_llm_component",
        variant_key="llama3.1",
        component_type=WebAgentLLMComponent,
        component_config_type=WebAgentLLMConfig,
    )
    component_factory = ComponentFactory(registry=registry)

    components = component_factory.build_components(config_dict=config, components_model_type=WebAgentLLMInstantiationModel)
    return components.webAgent_llm_component_llama
