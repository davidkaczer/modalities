from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_saving_execution import CheckpointEntityType, CheckpointSavingExecutionABC
from modalities.exceptions import CheckpointingError


class FSDPCheckpointSaving(CheckpointSavingExecutionABC):
    def __init__(
        self,
        checkpoint_path: Path,
        experiment_id: str,
        global_rank: int,
    ):
        """
        Implementation of checkpointing to disc via FSDP

        Args:
            checkpoint_path (Path): folder path to the checkpoint
            experiment_id (str): ID of the experiment
            global_rank (int): global rank within the current process group
        """
        self.checkpoint_path = checkpoint_path
        self.global_rank = global_rank
        self.experiment_id = experiment_id

    def _save_checkpoint(self, model: FSDP, optimizer: Optimizer, train_step_id: int):
        # saving the model via FULL_STATE_DICT and checkpoint via FULL_OPTIM_STATE_DICT
        # TODO Need to check if LR schedulers also need checkpointing
        model_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            module=model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=model_save_policy,
            optim_state_dict_config=optim_save_policy,
        ):
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()  # this gets the optimizer state dict object for each rank
            optim_state_dict = FSDP.optim_state_dict(
                model=model, optim=optimizer, optim_state_dict=optimizer_state
            )  # all the state dicts of the different ranks are synchronized

        if self.global_rank == 0:
            # save model
            model_checkpoint_path = self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                train_step_id=train_step_id,
                entity_type=CheckpointEntityType.MODEL,
            )

            model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model_state, model_checkpoint_path)

            # save optimizer
            optimize_checkpoint_path = self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                train_step_id=train_step_id,
                entity_type=CheckpointEntityType.OPTIMIZER,
            )
            torch.save(optim_state_dict, optimize_checkpoint_path)
        # we need this barrier here, such that all processes exit this function at the same time
        # Since we run throughput measurements in the trainer, the non-checkpointing ranks would already
        # trigger the time measurement in the trainer and would then wait for the checkpointing rank,
        # leading to wrong throughput measurements.
        dist.barrier()

    def _delete_checkpoint(self, train_step_id: int):
        if self.global_rank != 0:
            return

        files_paths_to_delete = self._get_paths_to_delete(train_step_id=train_step_id)
        for full_path in files_paths_to_delete:
            if full_path.exists():
                # unlink removes the file
                full_path.unlink()
            else:
                raise CheckpointingError(f"Checkpoint {full_path} could not be removed. It does not exist!")
