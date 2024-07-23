# test_dataset.py
import pytest
import torch
from data.oxford_iiit import OxfordIIITPet
from datamodule import DataModule
from models.unet import UNet

@pytest.fixture
def config():
    return {
        "root_dir": "./data",
        "height": 224,
        "width": 224,
        "batch_size": 32,
        "compression_method": "max_pool",
        "expansion_method": "upsample"
    }

def test_oxford_iiit_pet_dataset(config):
    dataset = OxfordIIITPet(config)
    assert len(dataset) > 0
    sample = dataset[0]
    assert "image" in sample and "mask" in sample
    assert sample["image"].shape == (3, 224, 224)
    assert sample["mask"].shape == (1, 224, 224)

def test_datamodule(config):
    datamodule = DataModule(config, OxfordIIITPet)
    datamodule.setup()
    assert datamodule.train_dataloader() is not None
    assert datamodule.val_dataloader() is not None
    assert datamodule.test_dataloader() is not None

    batch = next(iter(datamodule.train_dataloader()))
    assert "image" in batch and "mask" in batch
    assert batch["image"].shape == (config["batch_size"], 3, 224, 224)
    assert batch["mask"].shape == (config["batch_size"], 1, 224, 224)
    

if __name__ == "__main__":
    pytest.main([__file__])