import os
from pathlib import Path

from modalities.config.component_factory import ComponentFactory
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
import pytest
import torch
from modalities.webagent_llm_part.utils import get_concatenated_model
from modalities.config.config import load_app_config_dict
from modalities.models.model import NNModel
from tests.conftest import _ROOT_DIR

import transformers
import modalities

@pytest.fixture()
def set_env():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


@pytest.fixture()
def config_file_path() -> Path:

    config_file_path = _ROOT_DIR / Path(
        "tests/webAgent/model_config.yaml"
    )
    return config_file_path


@pytest.fixture()
def config_dict_without_checkpoint_path(config_file_path: Path) -> dict:
    return load_app_config_dict(config_file_path=config_file_path)


def test_initialize_llm_model(set_env, config_dict_without_checkpoint_path: dict) -> NNModel:
    model = get_concatenated_model(config_dict_without_checkpoint_path)
    assert type(model.llm) == transformers.models.llama.modeling_llama.LlamaForCausalLM
    assert type(model.html_model) == modalities.models.huggingface.huggingface_model.HuggingFacePretrainedEncoderDecoderModel