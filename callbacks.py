import torch
import lightning as pl
from tqdm.notebook import tqdm


class ClassAccuracyLoss(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        print(
            f"\n Epoch: {trainer.current_epoch} | Train Loss: {trainer.callback_metrics['train_loss']:.5f} | Train Acc: {trainer.callback_metrics['train_acc']:.5f}"
        )

    def on_validation_epoch_end(self, trainer, *args, **kwargs):
        print(
            f"\n Epoch: {trainer.current_epoch} | Val Loss: {trainer.callback_metrics['val_loss']:.5f} | Val Acc: {trainer.callback_metrics['val_acc']:.5f}"
        )

