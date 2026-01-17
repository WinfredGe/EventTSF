import os
import shutil
import time
import wandb
from typing import Dict, Callable
from pathlib import Path
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from experiments.base_experiment import BaseExperiment
from algorithms.non_stationary_prediction_algorithm import NonStationPrediction
from data.textandts_custom_datamodule import TextAndTSCustomDataModule
from myutils.logger import WandbSetup
from myutils.progress_bar import MyTQDMProgressBar


class SequencePredictionExperiment(BaseExperiment):
    @property
    def compatible_algorithms(self) -> Dict[str, Callable]:
        return {
            "non_station_prediction_algorithm": NonStationPrediction,
        }

    def training(self):
        WandbSetup.setup_torch_performance()
        if self.logger is None:
            project_name = self.root_cfg.get("wandb", {}).get("project", "simple-model-demo")
            run_name = self.root_cfg.get("wandb", {}).get("run_name", None)
            WandbSetup.init_wandb(
                project_name=project_name,
                name=run_name,
                config={
                    "epochs": self.root_cfg.experiment.epochs,
                    "lr": self.root_cfg.algorithm.lr,
                    "batch_size": self.root_cfg.data.batch_size,
                },
                mode="offline"
            )
            if wandb.config:
                print("\nMerging wandb sweep parameters into experiment's root_cfg:")
                for key, value in dict(wandb.config).items():
                    if "." in key:
                        parts = key.split(".")
                        curr = self.root_cfg
                        for i, part in enumerate(parts):
                            if i == len(parts) - 1:
                                curr[part] = value
                            else:
                                if part not in curr or not isinstance(curr[part], (DictConfig, dict)):
                                    curr[part] = OmegaConf.create({})
                                curr = curr[part]
                    else:
                        self.root_cfg[key] = value
            print("\nExperiment's root_cfg after merging wandb.config:")
            print(OmegaConf.to_yaml(self.root_cfg))
            self.logger = WandbSetup.create_logger(project_name, log_model=True)
        data_module = TextAndTSCustomDataModule(self.root_cfg)
        self.algo = self._build_algo()
        checkpoint_dir = Path(self.logger.experiment.dir)
        if checkpoint_dir.exists():
            for item in checkpoint_dir.iterdir():
                if item.is_file() and item.suffix == '.ckpt':
                    item.unlink()
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(
                monitor='val_loss',
                filename='best_model-{epoch:02d}-{val_loss:.4f}',
                save_top_k=1,
                mode='min',
                save_weights_only=False,
                dirpath=checkpoint_dir,
                save_last=False,
                verbose=True,
            ),
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            MyTQDMProgressBar(),
        ]
        trainer = pl.Trainer(
            logger=self.logger,
            max_epochs=self.root_cfg.experiment.epochs,
            callbacks=callbacks,
            accelerator='auto',
            log_every_n_steps=5,
            check_val_every_n_epoch=5,
            num_sanity_val_steps=0,
            enable_progress_bar=True,
            enable_model_summary=False,
        )
        print(f"Starting training for {self.root_cfg.experiment.epochs} epochs...")
        trainer.fit(self.algo, datamodule=data_module, ckpt_path=self.ckpt_path)
        print("Training complete.")

    def test(self):
        print("Executing test task.")
        if self.logger is None:
            project_name = self.root_cfg.get("wandb", {}).get("project", "simple-model-demo")
            run_name = self.root_cfg.get("wandb", {}).get("run_name", f"test")
            WandbSetup.init_wandb(
                project_name=project_name,
                name=run_name,
                config={
                    "test_mode": True,
                    "batch_size": self.root_cfg.data.batch_size,
                    "checkpoint_path": self.ckpt_path if self.ckpt_path else "auto-detected"
                },
                mode="offline"
            )
            if wandb.config:
                print("\nMerging wandb sweep parameters into experiment's root_cfg:")
                for key, value in dict(wandb.config).items():
                    if "." in key:
                        parts = key.split(".")
                        curr = self.root_cfg
                        for i, part in enumerate(parts):
                            if i == len(parts) - 1:
                                curr[part] = value
                            else:
                                if part not in curr or not isinstance(curr[part], (DictConfig, dict)):
                                    curr[part] = OmegaConf.create({})
                                curr = curr[part]
                    else:
                        self.root_cfg[key] = value
            print("\nExperiment's root_cfg after merging wandb.config:")
            print(OmegaConf.to_yaml(self.root_cfg))
            self.logger = WandbSetup.create_logger(project_name, log_model=False)
        data_module = TextAndTSCustomDataModule(self.root_cfg)
        if self.ckpt_path:
            load_path = self.ckpt_path
            print(f"Testing model from provided checkpoint: {load_path}")
        else:
            if self.logger and hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'dir'):
                checkpoint_dir = Path(self.logger.experiment.dir)
                if checkpoint_dir.exists():
                    checkpoint_files = list(checkpoint_dir.glob("best_model-*.ckpt"))
                    if checkpoint_files:
                        load_path = str(checkpoint_files[0])
                        print(f"Testing model from checkpoint: {load_path}")
                    else:
                        last_ckpt = checkpoint_dir / "last.ckpt"
                        if last_ckpt.exists():
                            load_path = str(last_ckpt)
                            print(f"Testing model from last checkpoint: {load_path}")
                        else:
                            print("No checkpoint files found. Using the current model state.")
                            load_path = None
                else:
                    print("Checkpoint directory not found. Using the current model state.")
                    load_path = None
            else:
                print("Logger not available. Using the current model state.")
                load_path = None
        trainer = pl.Trainer(
            logger=self.logger,
            accelerator='auto',
            enable_progress_bar=True,
            enable_model_summary=False,
        )
        if load_path:
            model = NonStationPrediction.load_from_checkpoint(
                load_path,
                root_cfg=self.root_cfg
            )
            trainer.test(model, datamodule=data_module)
        else:
            if self.algo is None:
                print("No model available for testing. Please run training first.")
                return
            trainer.test(self.algo, datamodule=data_module)
        print("Test task completed. Check WandB for test visualizations.")

