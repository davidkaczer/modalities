import dataclasses
import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch.optim import Optimizer
from transformers import GPT2TokenizerFast

from llm_gym.__main__ import load_app_config_dict
from llm_gym.checkpointing.checkpointing import CheckpointingIF
from llm_gym.config.config import AppConfig
from llm_gym.dataloader.create_index import IndexGenerator
from llm_gym.dataloader.large_file_lines_reader import LargeFileLinesReader
from llm_gym.dataset_loader import LLMDataLoader
from llm_gym.evaluator import Evaluator
from llm_gym.logging_broker.publisher import MessagePublisher
from llm_gym.loss_functions import Loss
from llm_gym.models.model import NNModel
from llm_gym.trainer import Trainer

_ROOT_DIR = Path(__file__).parents[1]


@pytest.fixture
def dummy_packed_data_path(tmpdir) -> Path:
    data = b""
    header_size_in_bytes = 8
    int_size_in_bytes = 4
    tokens = list(range(20))
    data += (len(tokens) * int_size_in_bytes).to_bytes(header_size_in_bytes, byteorder="big")
    data += b"".join([t.to_bytes(int_size_in_bytes, byteorder="big") for t in tokens])
    index = [(4, 24), (28, 40), (68, 12), (80, 4)]  # [(index,len), ...] -> in 4 bytes #lengths: 6,10,3,1
    data += pickle.dumps(index)
    dummy_packed_data_path = Path(tmpdir, "dummy.pbin")
    dummy_packed_data_path.write_bytes(data)
    return dummy_packed_data_path


@pytest.fixture
def dummy_config(monkeypatch) -> AppConfig:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    dummy_config_path = _ROOT_DIR / Path("config_files/config_lorem_ipsum.yaml")
    config_dict = load_app_config_dict(dummy_config_path)
    return AppConfig.model_validate(config_dict)


@dataclasses.dataclass
class DataPathCollection:
    raw_data_path: Path
    index_path: Path


@pytest.fixture
def dummy_data_path(tmpdir) -> DataPathCollection:
    source_raw_dummy_data_path = _ROOT_DIR / Path("./data/lorem_ipsum.jsonl")
    dummy_data_path = Path(tmpdir, source_raw_dummy_data_path.name)
    dummy_data_path.write_text(source_raw_dummy_data_path.read_text())
    index_path = LargeFileLinesReader.default_index_path(dummy_data_path)
    index_path.unlink(missing_ok=True)
    return DataPathCollection(raw_data_path=dummy_data_path, index_path=index_path)


@pytest.fixture
def indexed_dummy_data_path(dummy_data_path) -> DataPathCollection:
    index_generator = IndexGenerator(dummy_data_path.raw_data_path)
    index_generator.run(dummy_data_path.index_path)
    return dummy_data_path


@pytest.fixture
def gpt2_tokenizer() -> GPT2TokenizerFast:
    default_gpt2_tokenizer_path = Path(__file__).parents[1] / Path("data", "tokenizer", "tokenizer.json")
    assert default_gpt2_tokenizer_path.is_file()
    return GPT2TokenizerFast(tokenizer_file=str(default_gpt2_tokenizer_path))


@pytest.fixture(scope="function")
def checkpointing_mock():
    return MagicMock(spec=CheckpointingIF)


@pytest.fixture(scope="function")
def evaluator_mock():
    return MagicMock(spec=Evaluator)


@pytest.fixture(scope="function")
def nn_model_mock():
    return MagicMock(spec=NNModel)


@pytest.fixture(scope="function")
def optimizer_mock():
    return MagicMock(spec=Optimizer)


@pytest.fixture(scope="function")
def loss_mock():
    return MagicMock(spec=Loss, return_value=torch.rand(1, requires_grad=True))


@pytest.fixture(scope="function")
def llm_data_loader_mock():
    return MagicMock(spec=LLMDataLoader)


@pytest.fixture(scope="function")
def progress_publisher_mock():
    return MagicMock(spec=MessagePublisher)


@pytest.fixture(scope="function")
def trainer(progress_publisher_mock):
    return Trainer(
        local_rank=os.getenv("LOCAL_RANK"),
        batch_progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
    )


def set_env_cpu(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")

    # cleanup
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    # gloo for CPU testing with reduce
    torch.distributed.init_process_group(backend="gloo")
