import os
import torch
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from typing import Optional


class WandbSetup:

    @staticmethod
    def init_wandb(project_name: str, name: str, config: dict, mode: str = "offline") -> None:

        if wandb.run is not None:
            wandb.finish()

        wandb.init(
            project=project_name,
            name=name,
            config=config,
            dir=os.getcwd(),
            mode=mode
        )

    @staticmethod
    def create_logger(project_name: str, log_model: bool = True) -> WandbLogger:

        return WandbLogger(
            project=project_name,
            log_model=log_model,
        )

    @staticmethod
    def finish_wandb() -> None:

        if wandb.run is not None:
            wandb.finish()

    @staticmethod
    def setup_torch_performance():

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('medium')