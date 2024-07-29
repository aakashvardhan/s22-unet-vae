from typing import Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from data.oxford_iiit import OxfordIIITPet
torch.manual_seed(1)


class DataModule(LightningDataModule):
    """
    LightningDataModule subclass for handling data loading and processing.

    Args:
        config (dict): Configuration parameters for the data module.
        dataset: The dataset class to be used for loading the data.

    Attributes:
        config (dict): Configuration parameters for the data module.
        dataset: The dataset class to be used for loading the data.
        train_data (Optional[torch.utils.data.Dataset]): Training dataset.
        val_data (Optional[torch.utils.data.Dataset]): Validation dataset.
        test_data (Optional[torch.utils.data.Dataset]): Test dataset.
        train_args (dict): Arguments for training data loader.
        val_args (dict): Arguments for validation data loader.
    """

    def __init__(self, config, dataset=OxfordIIITPet):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.save_hyperparameters()
        self.train_data: Optional[torch.utils.data.Dataset] = None
        self.val_data: Optional[torch.utils.data.Dataset] = None
        self.test_data: Optional[torch.utils.data.Dataset] = None
        self.train_args = {
            "batch_size": self.config.batch_size,
            "shuffle": True,
            "num_workers": 4
        }

        self.val_args = {
            "batch_size": self.config.batch_size,
            "shuffle": False,
            "num_workers": 4
        }

    def setup(self, stage=None):
        """
        Setup method to prepare the data.

        Args:
            stage (str, optional): Stage of the training process. Defaults to None.
        """
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
        """
        Returns the training data loader.

        Returns:
            torch.utils.data.DataLoader: Training data loader.
        """
        return DataLoader(self.train_data, **self.train_args)

    def val_dataloader(self):
        """
        Returns the validation data loader.

        Returns:
            torch.utils.data.DataLoader: Validation data loader.
        """
        return DataLoader(self.val_data, **self.val_args)

    def test_dataloader(self):
        """
        Returns the test data loader.

        Returns:
            torch.utils.data.DataLoader: Test data loader.
        """
        return DataLoader(self.test_data, **self.val_args)
