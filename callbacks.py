import torch
import lightning as pl
from tqdm.notebook import tqdm
from utils.plot_example import plot_test_example


class ClassAccuracyLoss(pl.Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        print(
            f"\n Epoch: {trainer.current_epoch} | Train Loss: {trainer.callback_metrics['train_loss']:.5f} | Train Acc: {trainer.callback_metrics['train_acc']:.5f}"
        )

    def on_validation_epoch_end(self, trainer, *args, **kwargs):
        print(
            f"\n Epoch: {trainer.current_epoch} | Val Loss: {trainer.callback_metrics['val_loss']:.5f} | Val Acc: {trainer.callback_metrics['val_acc']:.5f}"
        )


class PlotExampleCallback(pl.Callback):
    def __init__(self, config, interval=5):
        super().__init__()
        self.config = config
        self.interval = interval

    def on_validation_epoch_end(self, trainer, *args, **kwargs):
        epoch = trainer.current_epoch
        if epoch % self.interval == 0:
            print(f"\n Plotting example image from validation set at epoch {epoch}")
            plot_test_example(trainer.datamodule.val_dataloader())
