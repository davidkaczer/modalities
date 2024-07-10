from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List

import torch.nn as nn
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_saving_instruction import CheckpointingInstruction


class CheckpointEntityType(Enum):
    MODEL = "model"
    OPTIMIZER = "optimizer"


class CheckpointSavingExecutionABC(ABC):
    CHECKPOINT_STRUCTURE = "eid_{experiment_id}-{entity}-num_steps_{num_train_steps}.bin"

    @abstractmethod
    def _save_checkpoint(self, model: nn.Module, optimizer: Optimizer, train_step_id: int):
        raise NotImplementedError

    @abstractmethod
    def _delete_checkpoint(self, train_step_id: int):
        raise NotImplementedError

    def run_checkpoint_instruction(
        self,
        checkpointing_instruction: CheckpointingInstruction,
        train_step_id: int,
        model: nn.Module,
        optimizer: Optimizer,
    ):
        if checkpointing_instruction.save_current:
            self._save_checkpoint(model=model, optimizer=optimizer, train_step_id=train_step_id)

        for train_step_id in checkpointing_instruction.checkpoints_to_delete:
            self._delete_checkpoint(train_step_id=train_step_id)

    def _get_checkpointing_path(
        self,
        experiment_id: str,
        train_step_id: int,
        entity_type: CheckpointEntityType,
    ) -> Path:
        entity_file_name = self.CHECKPOINT_STRUCTURE.format(
            experiment_id=experiment_id, entity=entity_type.value, num_train_steps=str(train_step_id + 1)
        )

        full_path = Path(self.checkpoint_path, experiment_id, entity_file_name)
        return full_path

    def _get_paths_to_delete(self, train_step_id: int) -> List[Path]:
        return [
            self._get_checkpointing_path(
                experiment_id=self.experiment_id, entity_type=entity_type, train_step_id=train_step_id
            )
            for entity_type in CheckpointEntityType
        ]
