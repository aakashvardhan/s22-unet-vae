import os
from typing import Optional

import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.oxford_iiit import OxfordIIITPet


class DataModule(LightningDataModule):
    def __init__(self, config, sep: str = " ", pin_memory: bool = True):
        super().__init__()
        self.config = config
        self.sep = sep
        self.pin_memory = pin_memory
        self.train_dataset: Optional[OxfordIIITPet] = None
        self.val_dataset: Optional[OxfordIIITPet] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._create_dataset(self.config["train_f"])
            self.val_dataset = self._create_dataset(self.config["val_f"])

            print(f"Number of training samples: {len(self.train_dataset)}")
            print(f"Number of validation samples: {len(self.val_dataset)}")

    def train_dataloader(self):
        return self._create_dataloader(
            self.train_dataset, self.config["batch_size"], shuffle=True
        )

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, 1, shuffle=False)

    def _create_dataset(self, data_file: str) -> OxfordIIITPet:
        df_list = pd.read_csv(data_file, sep=self.sep, header=None)[0].to_list()

        img_list = [os.path.join(self.config["img_dir"], f"{i}.jpg") for i in df_list]
        mask_list = [os.path.join(self.config["mask_dir"], f"{i}.png") for i in df_list]

        return OxfordIIITPet(
            imgs_file=img_list,
            masks_file=mask_list,
            transform_img=None,
            transform_mask=None,
        )

    def _create_dataloader(
        self, dataset: OxfordIIITPet, batch_size: int, shuffle: bool
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config["num_workers"],
            pin_memory=self.pin_memory,
        )
