import json
from dataclasses import dataclass,fields
from typing import Any, Dict


@dataclass
class UNetConfig:
    # Model parameters
    in_channels: int = 3
    out_channels: int = 3
    num_filters: int = 64
    num_layers: int = 4

    # Training parameters
    batch_size: int = 16
    learning_rate: float = 0.001
    epochs: int = 25

    # Data parameters
    root_dir: str = "./data"
    height: int = 240
    width: int = 240

    # Optimizer and loss function parameters
    optimizer: str = "adam"
    loss_function: str = "cross_entropy"

    # Encoder and Decoder parameters
    channel_reduction_method: str = "max_pool"
    channel_expansion_method: str = "upsample"

    softmax_dim: int = 1

    # Any other hyperparameters
    dropout_rate: float = 0.5
    augmentation: bool = True

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

def update_config(config: UNetConfig, json_data: dict):
    for field in fields(config):
        if field.name in json_data:
            setattr(config, field.name, json_data[field.name])
    return config

# if __name__ == "__main__":
#     config = UNetConfig()
#     json_data = load_config("training_1.json")
#     config = update_config(config, json_data)
#     print(config)