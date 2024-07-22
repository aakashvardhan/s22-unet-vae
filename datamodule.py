from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, random_split
from typing import Optional

torch.manual_seed(1)


class DataModule(LightningDataModule):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.save_hyperparameters(self.config)
        self.train_data: Optional[torch.utils.data.Dataset] = None
        self.val_data: Optional[torch.utils.data.Dataset] = None
        self.test_data: Optional[torch.utils.data.Dataset] = None
        self.train_args = {
            "batch_size": self.config['batch_size'],
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
        }

        self.val_args = {
            "batch_size": 10,
            "shuffle": False,
            "num_workers": 4,
            "pin_memory": True,
        }

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_data = self.dataset(self.config, split="trainval")

            self.test_data = self.dataset(self.config, split="test")

            total_len = len(train_data)
            train_len = int(0.8 * total_len)
            val_len = total_len - train_len

            self.train_data, self.val_data = random_split(
                train_data,
                [train_len, val_len],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, **self.train_args)

    def val_dataloader(self):
        return DataLoader(self.val_data, **self.val_args)

    def test_dataloader(self):
        return DataLoader(self.test_data, **self.val_args)
