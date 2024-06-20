from pathlib import Path

import pytest
import torch

from modalities.__main__ import Main, load_app_config_dict
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.running_env.cuda_env import CudaEnv
from tests.conftest import _ROOT_DIR


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
def test_e2e_anymal_training_run_without_checkpoint(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")

    # Load config
    dummy_config_path = _ROOT_DIR / Path("tests/models/anymal/config_example_anymal_vision.yaml")
    config_dict = load_app_config_dict(dummy_config_path)

    # Disable checkpointing
    config_dict["checkpoint_saving"]["config"]["checkpoint_saving_strategy"]["config"]["k"] = 0

    main = Main(config_dict, dummy_config_path)
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        components = main.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main.run(components)
