import pytest
import torch.cuda

from modalities.__main__ import Main


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
def test_e2e_training_run_wout_ckpt(monkeypatch, dummy_config, dummy_config_path):
    # patch in env variables
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")
    main = Main(dummy_config, dummy_config_path)
    main.run()
