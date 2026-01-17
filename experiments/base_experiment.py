from typing import Optional, Union, Dict, Callable
from omegaconf import DictConfig
import pathlib
from abc import ABC, abstractmethod
from lightning.pytorch.loggers.wandb import WandbLogger


class BaseExperiment(ABC):
    @property
    @abstractmethod
    def compatible_algorithms(self) -> Dict[str, Callable]:
        raise NotImplementedError(
            "Subclasses must define 'compatible_algorithms' as a dictionary mapping algorithm names to constructors."
        )

    def __init__(
            self,
            root_cfg: DictConfig,
            logger: Optional[WandbLogger] = None,
            ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        super().__init__()
        self.root_cfg = root_cfg
        self.logger = logger
        self.ckpt_path = ckpt_path
        self.algo = None

    def _build_algo(self):
        algo_name = self.root_cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this Experiment class. "
                "Make sure you define compatible_algorithms correctly and make sure that each key has "
                "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"
            )
        try:
            return self.compatible_algorithms[algo_name](self.root_cfg)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate algorithm '{algo_name}' using compatible_algorithms."
            ) from e

    def exec_task(self, task: str) -> None:
        if hasattr(self, task) and callable(getattr(self, task)):
            getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )

