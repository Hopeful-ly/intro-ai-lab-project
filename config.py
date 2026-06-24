from dataclasses import dataclass, field

import numpy as np

# MODES: "lab_regression", "lab_classification", "rgb_regression"


@dataclass
class ModelConfig:
    mode: str = "lab_classification"
    base_channels: int = 72  # width: number of filters in the first conv block
    depth: int = 3  # how many down/up-sampling levels (32 -> 32/2**depth)
    grid_size: int = (
        16  # ab-bins per axis for "lab_classification" (grid_size**2 classes)
    )
    decode_temperature: float = 0.30

    residual: bool = True
    dropout: float = 0.1
    dilated_bottleneck: bool = True

    class_rebalance: bool = True
    rebalance_lambda: float = 1


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 40
    lr: float = 1e-4
    optimizer: str = "sgd"  # "adam" | "sgd"
    weight_decay: float = 0.001187
    momentum: float = 0.99  # only used by sgd
    seed: int = 42
    val_fraction: float = 0.1
    test_fraction: float = 0.05
    num_workers: int = 2
    data_root: str = "./data"
    amp: bool = True  # mixed precision (only activates on cuda)

    datasets: tuple = ("stl10", "places365")
    image_size: int = 96
    max_images_per_dataset: int | None = 50000  # cap each source (None = use all)
    places365_split: str = (
        "train-standard"  # "val" (~36k, ~2GB) or "train-standard" (~1.8M, large)
    )
    min_colorfulness: float = 0.0  # filters not-colorful images

    augment: bool = True
    aug_min_scale: float = 0.5

    checkpoint_every: int = 10
    checkpoint_dir: str = "checkpoints"
    resume_from: str | None = None


@dataclass
class SearchConfig:
    method: str = "random"
    n_trials: int = 50
    search_epochs: int = 8
    subset_size: int | None = 10000
    seed: int = 0
    space: dict = field(
        default_factory=lambda: {
            "mode": [
                "lab_regression",
                "lab_classification",
                "rgb_regression",
            ],
            "base_channels": np.arange(32, 81, 8).tolist(),
            "depth": list(range(3, 6)),
            "lr": np.round(np.logspace(np.log10(3e-4), np.log10(1e-2), 16), 6).tolist(),
            "batch_size": np.arange(16, 49, 8).tolist(),
            "optimizer": ["adam", "sgd"],
            "momentum": np.round(np.linspace(0.5, 0.99, 10), 3).tolist(),
            "weight_decay": np.round(
                np.logspace(np.log10(1e-4), np.log10(3e-3), 12), 6
            ).tolist(),
            "dropout": np.round(np.linspace(0.0, 0.3, 13), 3).tolist(),
            "rebalance_lambda": np.round(np.linspace(0.0, 1.0, 11), 2).tolist(),
            "aug_min_scale": np.round(np.linspace(0.3, 0.8, 6), 2).tolist(),
        }
    )
