import json
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class UNetConfig:
    # Model parameters
    in_channels: int = 3
    out_channels: int = 1
    num_filters: int = 64
    num_layers: int = 4

    # Training parameters
    batch_size: int = 16
    learning_rate: float = 0.001
    epochs: int = 50

    # Data parameters
    root_dir: str = "./data"
    height: int = 240
    width: int = 240

    # Optimizer and loss function parameters
    optimizer: str = "adam"
    loss_function: str = "cross_entropy"

    # Any other hyperparameters
    dropout_rate: float = 0.5
    augmentation: bool = True

    def load_from_file(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as file:
            config_dict = json.load(file)
            self.update_from_dict(config_dict)

    def save_to_file(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.__dict__, file, indent=4)

    def update_from_dict(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
