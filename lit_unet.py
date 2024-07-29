import torch
import torch.nn as nn
import torchmetrics
from lightning import LightningModule

from models.unet import UNet
import random
import matplotlib.pyplot as plt
import numpy as np
from utils.multiclass_dice_loss import DiceLoss
import wandb


class LitUNet(LightningModule):
    def __init__(self, config, learning_rate=1e-3, best_lr=1e-3):

        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.best_lr = best_lr
        self.model = UNet(self.config)

        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=3
        )
        self.valid_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=3
        )

        if self.config.loss_function == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss()
        elif self.config.loss_function == "dice":
            self.loss_function = DiceLoss(self.config)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.best_lr,
                steps_per_epoch=int(len(self.trainer.datamodule.train_dataloader())),
                epochs=self.config.epochs,
            ),
            "interval": "step",
            "frequency": 1,
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        y = y.squeeze(1).to(dtype=torch.long)
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)

        # Log training metrics
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        y = y.squeeze(1).to(dtype=torch.long)
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)

        # Log validation metrics
        self.valid_acc(y_hat, y)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True)

        # Log validation loss
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        
        self.x = x
        self.y = y

    def on_validation_epoch_end(self):
        # Get a batch of validation data
        
        x_, y_ = self.x.to(self.device), self.y.to(self.device)

        # Make predictions
        with torch.no_grad():
            predictions = self(x_)

        # Create the plot
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        for i in range(3):
            # Original Image
            axes[i, 0].imshow(x_[i].cpu().permute(1, 2, 0).numpy())
            axes[i, 0].set_title(f"Original Image {i+1}")
            axes[i, 0].axis("off")

            # Original Mask
            axes[i, 1].imshow(y_[i].cpu().numpy(), cmap="gray")
            axes[i, 1].set_title(f"Original Mask {i+1}")
            axes[i, 1].axis("off")

            # Predicted Mask
            pred_mask = torch.argmax(predictions[i], dim=0).cpu().numpy()
            axes[i, 2].imshow(pred_mask, cmap="gray")
            axes[i, 2].set_title(f"Predicted Mask {i+1}")
            axes[i, 2].axis("off")

        plt.tight_layout()

        # # Log the figure to WandB
        self.logger.experiment.log({"validation_samples": wandb.Image(plt)})

        # Close the figure to free up memory
        plt.close(fig)

        plt.tight_layout()
        plt.savefig("asset")
        print(f"Sample output images are saved at {'asset'}")


# Test the implementation

if __name__ == "__main__":
    from lightning.pytorch import Trainer, seed_everything
    from config import UNetConfig, load_config, update_config
    import torch
    import os
    from lit_unet import LitUNet
    from datamodule import DataModule
    import lightning as pl

    config = UNetConfig()
    json_data = load_config("training_1.json")
    config = update_config(config, json_data)
    print(config)

    config.batch_size = 32
    print(config)

    data_module = DataModule(config)
    data_module.setup()

    model = LitUNet(config)

    trainer = pl.Trainer(fast_dev_run=True, accelerator="gpu")

    trainer.fit(model=model, datamodule=data_module)
