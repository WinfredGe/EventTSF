import hydra
from omegaconf import DictConfig, OmegaConf
from experiments.sequence_prediction_experiment import SequencePredictionExperiment
from myutils.logger import WandbSetup


@hydra.main(version_base=None, config_path="configs", config_name="hydra_config")
def main(cfg: DictConfig) -> None:
    try:
        OmegaConf.set_struct(cfg, False)
        print(f"Metadata path: {cfg.algorithm.metadata_path}")
        checkpoint_path = None
        experiment = SequencePredictionExperiment(root_cfg=cfg, ckpt_path=checkpoint_path)
        experiment.exec_task("training")
        experiment.exec_task("test")

    finally:
        WandbSetup.finish_wandb()


if __name__ == '__main__':
    main()