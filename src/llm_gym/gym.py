
from typing import List
from llm_gym.checkpointing.checkpointing import Checkpointing
from llm_gym.dataset_loader import LLMDataLoader
from llm_gym.evaluator import Evaluator
from llm_gym.loss_functions import Loss
from llm_gym.trainer import Trainer
from llm_gym.forward_pass import ModelInferenceComponent
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler


class Gym:
    def __init__(self, checkpointing: Checkpointing, trainer: Trainer, evaluator: Evaluator,
                 model_inference_component: ModelInferenceComponent, optimizer: Optimizer,
                 loss_fun: Loss) -> None:
        self.checkpointing = checkpointing
        self.trainer = trainer
        self.evaluator = evaluator
        self.model_inference_component = model_inference_component
        self.optimizer = optimizer
        self.loss_fun = loss_fun

    def run(self, num_epochs: int, train_data_loader: LLMDataLoader, evaluation_data_loaders: List[LLMDataLoader],
            sampler: DistributedSampler = None):
        for current_epoch in range(num_epochs):
            if sampler is not None:
                # In distributed mode, calling the set_epoch() method at the beginning of each epoch before creating the DataLoader
                # iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will
                # be always used.
                # see discussion: https://discuss.pytorch.org/t/why-is-sampler-set-epoch-epoch-needed-for-distributedsampler/149672
                sampler.set_epoch(current_epoch)

            train_result = self.trainer.train_epoch(model_inference_component=self.model_inference_component,
                                                    train_loader=train_data_loader,
                                                    loss_fun=self.loss_fun, optimizer=self.optimizer)
            eval_result = self.evaluator.evaluate_epoch(model_inference_component=self.model_inference_component,
                                                        data_loaders=evaluation_data_loaders, loss_fun=self.loss_fun)

            # TODO: implement early stopping
            self.checkpointing.run(num_epochs=num_epochs, current_epoch=current_epoch, evaluation_result=eval_result,
                                   model=self.model_inference_component.model,  # TODO self.model_inference_component.model is a bit ugly
                                   early_stoppping_criterion_fulfilled=False)
