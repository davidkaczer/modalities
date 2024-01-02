from llm_gym.checkpointing.checkpointing import (
    Checkpointing,
    CheckpointingExecutionIF,
    CheckpointingIF,
    CheckpointingStrategyIF,
)
from llm_gym.config.config import CheckpointingConfig
from llm_gym.fsdp.fsdp_running_env import RunningEnv
from llm_gym.resolver_register import ResolverRegister


class CheckpointingFactory:
    @staticmethod
    def get_checkpointing(
        resolvers: ResolverRegister,
        config: CheckpointingConfig,
        running_env: RunningEnv,
        experiment_id: str,
        num_ranks: int,
    ) -> CheckpointingIF:
        checkpointing_strategy: CheckpointingStrategyIF = resolvers.build_component_by_config(
            config=config.checkpointing_strategy, extra_kwargs={}
        )

        checkpointing_execution: CheckpointingExecutionIF = resolvers.build_component_by_config(
            config=config.checkpointing_execution,
            extra_kwargs={"experiment_id": experiment_id, "model_wrapping_fn": running_env.wrap_model},
        )

        checkpointing = Checkpointing(
            checkpointing_strategy=checkpointing_strategy,
            checkpointing_execution=checkpointing_execution,
            num_ranks=num_ranks,
        )

        return checkpointing