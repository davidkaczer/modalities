import os
from pathlib import Path

import pytest
import transformers

import modalities
from modalities.config.config import load_app_config_dict
from modalities.models.model import NNModel
from modalities.models.utils import ModelTypeEnum, get_model_from_config
from tests.conftest import _ROOT_DIR


@pytest.fixture()
def set_env():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


@pytest.fixture()
def config_file_path() -> Path:
    config_file_path = _ROOT_DIR / Path("config_files/training/config_lorem_ipsum_span_masking_T5.yaml")
    return config_file_path


@pytest.fixture()
def config_dict_without_checkpoint_path(config_file_path: Path) -> dict:
    return load_app_config_dict(config_file_path=config_file_path)


def test_initialize_llm_model(set_env, config_dict_without_checkpoint_path: dict) -> NNModel:
    model = get_model_from_config(config=config_dict_without_checkpoint_path, model_type=ModelTypeEnum.WEBAGENT_LLM)
    assert isinstance(model.llm, transformers.models.llama.modeling_llama.LlamaForCausalLM)
    assert isinstance(
        model.html_model, modalities.models.huggingface.huggingface_model.HuggingFacePretrainedEncoderDecoderModel
    )

    for name, param in model.llm.named_parameters():
        assert not param.requires_grad
    for name, param in model.html_model.named_parameters():
        assert param.requires_grad
