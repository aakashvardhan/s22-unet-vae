import os
import json
from argparse import ArgumentParser
from typing import Dict, Any

import torch
import wandb

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner

from callbacks import ClassAccuracyLoss
from config import UNetConfig, update_config
from datamodule import DataModule
from lit_unet import LitUNet


def parse_args() -> Dict[str, Any]:
    parser = ArgumentParser(description="UNet Training Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for the data loader (overrides config file)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train the model (overrides config file)",
    )
    return vars(parser.parse_args())


def setup_wandb(config: UNetConfig) -> WandbLogger:
    os.environ["WANDB_NOTEBOOK_NAME"] = "./notebooks/train-unet-model-1.ipynb"
    wandb.init(settings=wandb.Settings(_service_wait=300))
    logger = WandbLogger(project="s22-unet")
    logger.experiment.config.update(config.__dict__)
    return logger


def setup_callbacks(config: UNetConfig) -> list:
    return [
        ModelCheckpoint(
            dirpath="checkpoints/",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            filename="model-{epoch:02d}-{val_loss:.2f}-{val_loss:4f}",
            save_last=True,
            verbose=True,
        ),
        ClassAccuracyLoss(),
        LearningRateMonitor(logging_interval="step", log_momentum=True),
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        RichModelSummary(max_depth=2),
    ]


def setup_trainer(config: UNetConfig, logger: WandbLogger, callbacks: list) -> Trainer:
    return Trainer(
        precision="16-mixed",
        max_epochs=config.epochs,
        accelerator="cuda",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        check_val_every_n_epoch=3,
        num_sanity_val_steps=2,
    )


def find_best_lr(trainer: Trainer, model: LitUNet, data_module: DataModule) -> float:
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        model,
        datamodule=data_module,
        min_lr=1e-4,
        max_lr=1,
        num_training=trainer.max_epochs,
        attr_name="learning_rate",
    )
    return lr_finder.suggestion()


def main():
    args = parse_args()

    try:
        with open(args["config"], "r") as config_file:
            json_config = json.load(config_file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading configuration file: {e}")
        return

    config = UNetConfig()
    config = update_config(config, json_config)

    if args["batch_size"]:
        config.batch_size = args["batch_size"]
    if args["epochs"]:
        config.epochs = args["epochs"]

    seed_everything(42, workers=True)
    torch.cuda.empty_cache()

    print("Setting up DataModule...")
    data_module = DataModule(config)
    data_module.setup()
    print("DataModule setup complete.")

    print("Setting up UNet model...")
    model = LitUNet(config)

    logger = setup_wandb(config)
    callbacks = setup_callbacks(config)
    trainer = setup_trainer(config, logger, callbacks)

    print("Finding the best learning rate...")
    new_lr = find_best_lr(trainer, model, data_module)
    print(f"Best learning rate: {new_lr}")
    model.learning_rate = new_lr

    print("Training the model...")
    trainer.fit(model=model, datamodule=data_module)

    print("Evaluating the model...")
    trainer.validate(model=model, datamodule=data_module)
    print("Model evaluation complete.")


if __name__ == "__main__":
    main()
