import glob
import os
import sys

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from config import TrainConfig

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm")


def _to_rgb(img):
    return img.convert("RGB")  # normalise grayscale/CMYK/RGBA -> 3 channels


def make_transforms(image_size: int, train: bool, aug_min_scale: float = 0.5):
    if train:
        return T.Compose(
            [
                T.Lambda(_to_rgb),
                T.RandomResizedCrop(
                    image_size, scale=(aug_min_scale, 1.0), antialias=True
                ),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
    return T.Compose(
        [
            T.Lambda(_to_rgb),
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
    )


def _build_source(name: str, transform, cfg: TrainConfig):
    root = cfg.data_root
    try:
        if name == "cifar10":  # @TODO: delete this.
            print("don't use cifar10; use other ones...")
            return torchvision.datasets.CIFAR10(
                root, train=True, download=True, transform=transform
            )
        if name == "stl10":
            return torchvision.datasets.STL10(
                root, split="unlabeled", download=True, transform=transform
            )
        if name == "places365":
            return torchvision.datasets.Places365(
                root,
                split=cfg.places365_split,
                small=True,
                download=True,
                transform=transform,
            )
    except Exception as e:  # noqa: BLE001
        print(
            f"[data] WARNING: could not load source {name!r}: {type(e).__name__}: {e}"
        )
        return None
    raise ValueError(f"unknown dataset source: {name!r}")


def _capped(ds, max_n, seed):
    """Deterministically cap a dataset to at most ``max_n`` images."""
    if max_n is None or len(ds) <= max_n:
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:max_n].tolist()
    return Subset(ds, idx)


@torch.no_grad()
def _colourful_indices(pool, min_cf, batch_size=512):
    """Indices of images whose colourfulness >= ``min_cf`` (drops near-grayscale)."""
    keep = []
    loader = DataLoader(pool, batch_size=batch_size, shuffle=False, num_workers=0)
    seen = 0
    for rgb, _ in loader:
        R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        rg = (R - G).flatten(1)
        yb = (0.5 * (R + G) - B).flatten(1)
        cf = torch.sqrt(rg.std(1) ** 2 + yb.std(1) ** 2) + 0.3 * torch.sqrt(
            rg.mean(1) ** 2 + yb.mean(1) ** 2
        )
        for j, ok in enumerate(cf >= min_cf):
            if ok:
                keep.append(seen + j)
        seen += rgb.size(0)
    return keep


def get_datasets(cfg: TrainConfig):
    train_tf = make_transforms(
        cfg.image_size, train=cfg.augment, aug_min_scale=cfg.aug_min_scale
    )
    clean_tf = make_transforms(cfg.image_size, train=False)

    aug_sources, clean_sources = [], []
    for i, name in enumerate(cfg.datasets):
        aug = _build_source(name, train_tf, cfg)
        clean = _build_source(name, clean_tf, cfg)
        if aug is None or clean is None:
            continue
        seed = cfg.seed + i  # same subset selected for aug and clean copies
        aug_sources.append(_capped(aug, cfg.max_images_per_dataset, seed))
        clean_sources.append(_capped(clean, cfg.max_images_per_dataset, seed))
        print(f"[data] {name}: {len(aug_sources[-1])} images")

    if not aug_sources:
        raise RuntimeError(f"no usable datasets from {cfg.datasets}")

    aug_pool = ConcatDataset(aug_sources)
    clean_pool = ConcatDataset(clean_sources)

    valid = list(range(len(aug_pool)))
    if cfg.min_colorfulness > 0:
        print(
            f"[data] filtering near-grayscale images (cf < {cfg.min_colorfulness}) ..."
        )
        valid = _colourful_indices(clean_pool, cfg.min_colorfulness)
        print(f"[data] kept {len(valid)}/{len(aug_pool)} colourful images")

    g = torch.Generator().manual_seed(cfg.seed)
    perm = [valid[k] for k in torch.randperm(len(valid), generator=g).tolist()]
    n_val = int(len(perm) * cfg.val_fraction)
    n_test = int(len(perm) * cfg.test_fraction)
    val_idx = perm[:n_val]
    test_idx = perm[n_val : n_val + n_test]
    train_idx = perm[n_val + n_test :]

    train = Subset(aug_pool, train_idx)
    val = Subset(clean_pool, val_idx)
    test = Subset(clean_pool, test_idx)
    print(
        f"[data] image_size={cfg.image_size} | train={len(train)} val={len(val)} test={len(test)}"
    )
    return train, val, test


def make_loader(dataset, batch_size: int, shuffle: bool, device, num_workers: int = 2):
    workers = (
        0 if (device.type == "cpu" or sys.platform == "win32") else num_workers
    )  # this used to cause problems
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        # drop the ragged last batch only when it wouldn't empty the loader
        drop_last=shuffle and len(dataset) > batch_size,
        persistent_workers=(workers > 0),
    )
