import os.path
import signal
import time

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig

import src.utilities.config_utils as cfg_utils
from src.interface import get_model_and_data
from src.utilities.utils import divein, get_logger, melk


log = get_logger(__name__)


def run_model(config: DictConfig) -> float:
    r"""
    This function runs/trains/tests the model.

    .. note::
        It is recommended to call this function by running its underlying script, ``src.train.py``,
        as this will enable you to make the best use of the command line integration with Hydra.
        For example, you can easily train a UNet for 10 epochs on the CPU with:

        >>>  python train.py trainer.max_epochs=10 trainer.accelerator="cpu" model=unet_resnet callbacks=default

    Args:
        config: A DictConfig object generated by hydra containing the model, data, callbacks & trainer configuration.

    Returns:
        float: the best model score reached while training the model.
                E.g. "val/mse", the mean squared error on the validation set.
    """
    # Seed for reproducibility
    pl.seed_everything(config.seed)

    # If not resuming training, check if run already exists (with same hyperparameters and seed)
    config = cfg_utils.extras(config, if_wandb_run_already_exists="new")

    wandb_id = config.logger.wandb.get("id") if hasattr(config.logger, "wandb") else None
    uses_wandb = wandb_id is not None
    if uses_wandb and config.wandb_status == "resume":
        # Reload model checkpoint if needed to resume training
        # Set the checkpoint to reload
        ckpt_filename = config.get("ckpt_path")
        if config.get("test_mode"):
            # Use best model checkpoint for testing
            ckpt_filename = ckpt_filename or "best.ckpt"
        else:
            # Use last model checkpoint for resuming training
            if ckpt_filename is not None:
                log.warning(f'Checkpoint used to resume training is not "last.ckpt" but "{ckpt_filename}"!')
            else:
                ckpt_filename = "last.ckpt"

        # if os.path.exists(ckpt_filename) and wandb_id in ckpt_filename:
        if os.path.exists(ckpt_filename):
            # Load model checkpoint from local file. For safety, only do this if the wandb run id is in the path.
            ckpt_path = ckpt_filename
            log.info(f"Loading model weights from local checkpoint: {ckpt_path}.")
        else:
            ckpt_path = f"{wandb_id}-{ckpt_filename}"
            if not os.path.exists(ckpt_path):
                # Download model checkpoint from wandb
                ckpt_path2 = wandb.restore(ckpt_filename, run_path=wandb.run.path, replace=True, root=os.getcwd()).name
                os.rename(ckpt_path2, ckpt_path)
    else:
        ckpt_path = None

    # Obtain the instantiated model and data classes from the config
    model, datamodule = get_model_and_data(config)

    # Init Lightning callbacks and loggers (e.g. model checkpointing and Wandb logger)
    callbacks = cfg_utils.get_all_instantiable_hydra_modules(config, "callbacks")
    loggers = cfg_utils.get_all_instantiable_hydra_modules(config, "logger")

    # Init Lightning trainer
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=loggers)

    # Send some parameters from config to be saved by the lightning loggers
    cfg_utils.log_hyperparameters(
        config=config, model=model, data_module=datamodule, trainer=trainer, callbacks=callbacks
    )
    if config.get("test_mode"):
        pass
    else:
        if hasattr(signal, "SIGUSR1"):  # Windows does not support signals
            signal.signal(signal.SIGUSR1, melk(trainer, config.ckpt_dir))
            signal.signal(signal.SIGUSR2, divein(trainer))

        def fit(ckpt_filepath=None):
            trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_filepath)

        try:
            # Train the model
            fit(ckpt_filepath=ckpt_path)
            log.info(" ---------------- Training finished successfully ----------------")
        except Exception as e:
            melk(trainer, config.ckpt_dir)()
            raise e

    # Testing:
    if config.get("test_after_training") or config.get("test_mode"):
        if config.get("test_mode"):
            test_what = {"ckpt_path": ckpt_path, "model": model}
        else:
            test_what = {"ckpt_path": "best"} if hasattr(callbacks, "model_checkpoint") else {"model": model}
        trainer.test(datamodule=datamodule, **test_what)

    if uses_wandb:
        try:
            wandb.finish()
            log.info(" ---------------- Sleeping for 10 seconds to make sure wandb finishes... ----------------")
            time.sleep(10)
        except (FileNotFoundError, PermissionError) as e:
            log.info(f"Wandb finish error:\n{e}")

    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path and False:
        # This is how the best model weights can be reloaded back:
        model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            datamodule_config=config.datamodule,
        )

    # return best score (e.g. validation mse). This is useful when using Hydra+Optuna HP tuning.
    return trainer.checkpoint_callback.best_model_score


@hydra.main(config_path="configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig) -> float:
    """Run/train model based on the config file configs/main_config.yaml (and any command-line overrides)."""
    return run_model(config)


if __name__ == "__main__":
    main()
