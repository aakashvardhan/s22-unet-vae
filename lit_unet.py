import torch
import torch.nn as nn
import torchmetrics
from lightning import LightningModule

from models.unet import UNet
from utils.multiclass_dice_loss import DiceLoss


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
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        y = y.squeeze(1).to(dtype=torch.long)
        y_hat, _ = self(x)
        loss = self.loss_function(y_hat, y)

        # Log training metrics
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)

        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        
    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        y = y.squeeze(1).to(dtype=torch.long)
        y_hat, _ = self(x)
        loss = self.loss_function(y_hat, y)

        # Log validation metrics
        self.valid_acc(y_hat, y)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True)

        # Log validation loss
        self.log("valid_loss", loss, on_step=True, on_epoch=True)
