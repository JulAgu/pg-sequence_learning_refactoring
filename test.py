import hydra
from omegaconf import DictConfig

@hydra.main(config_path='conf',
            config_name='config',
            version_base='1.3')
def main(cfg: DictConfig) -> None:
  print(cfg.hyperparameters.embedding_dims)
  print(cfg.num_epochs)
  print(cfg.experiment.exp_name)

if __name__ == "__main__":
    main()