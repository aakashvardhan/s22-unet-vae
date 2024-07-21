import pytest
import os
import tempfile
import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from datamodule import DataModule  # Replace 'your_module' with the actual module name


@pytest.fixture
def mock_config():
    return {
        "train_f": "train.txt",
        "val_f": "val.txt",
        "img_dir": "images",
        "mask_dir": "masks",
        "batch_size": 32,
        "num_workers": 4,
    }


@pytest.fixture
def mock_data_files(tmp_path):
    # Create mock train and val files
    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "val.txt"

    train_data = ["img1", "img2", "img3"]
    val_data = ["img4", "img5"]

    pd.Series(train_data).to_csv(train_file, index=False, header=False)
    pd.Series(val_data).to_csv(val_file, index=False, header=False)

    # Create mock image and mask directories
    os.mkdir(tmp_path / "images")
    os.mkdir(tmp_path / "masks")

    # Update config with temporary paths
    mock_config = {
        "train_f": str(train_file),
        "val_f": str(val_file),
        "img_dir": str(tmp_path / "images"),
        "mask_dir": str(tmp_path / "masks"),
        "batch_size": 32,
        "num_workers": 4,
    }

    return mock_config


def test_datamodule_initialization(mock_config):
    data_module = DataModule(mock_config)
    assert isinstance(data_module, LightningDataModule)
    assert data_module.config == mock_config


def test_datamodule_setup(mock_data_files):
    data_module = DataModule(mock_data_files)
    data_module.setup()

    assert data_module.train_dataset is not None
    assert data_module.val_dataset is not None
    assert len(data_module.train_dataset) == 3
    assert len(data_module.val_dataset) == 2


def test_datamodule_dataloaders(mock_data_files):
    data_module = DataModule(mock_data_files)
    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(val_dataloader, DataLoader)
    assert train_dataloader.num_workers == mock_data_files["num_workers"]
    assert val_dataloader.num_workers == mock_data_files["num_workers"]


def test_datamodule_pin_memory(mock_data_files):
    data_module = DataModule(mock_data_files, pin_memory=True)
    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    assert train_dataloader.pin_memory
    assert val_dataloader.pin_memory

    data_module = DataModule(mock_data_files, pin_memory=False)
    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    assert not train_dataloader.pin_memory
    assert not val_dataloader.pin_memory
