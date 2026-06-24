import itertools
import random
from dataclasses import replace

import torch
from torch.utils.data import Subset

from config import ModelConfig, SearchConfig, TrainConfig
from engine import train_model


def _subset(ds, n, seed):
    if n is None or len(ds) <= n:
        return ds
    g = torch.Generator().manual_seed(seed)
    return Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())


def build_trials(scfg: SearchConfig) -> list[dict]:
    keys = list(scfg.space.keys())
    if scfg.method == "grid":
        combos = itertools.product(*(scfg.space[k] for k in keys))
        return [dict(zip(keys, combo)) for combo in combos]
    if scfg.method == "random":
        rng = random.Random(scfg.seed)
        return [
            {k: rng.choice(scfg.space[k]) for k in keys} for _ in range(scfg.n_trials)
        ]
    raise ValueError(f"unknown search method: {scfg.method!r} (use 'grid' or 'random')")


def apply_params(params: dict, base_mcfg: ModelConfig, base_tcfg: TrainConfig):
    mcfg = replace(
        base_mcfg,
        mode=params.get("mode", base_mcfg.mode),
        base_channels=params.get("base_channels", base_mcfg.base_channels),
        depth=params.get("depth", base_mcfg.depth),
        dropout=params.get("dropout", base_mcfg.dropout),
        rebalance_lambda=params.get("rebalance_lambda", base_mcfg.rebalance_lambda),
    )
    tcfg = replace(
        base_tcfg,
        lr=params.get("lr", base_tcfg.lr),
        batch_size=params.get("batch_size", base_tcfg.batch_size),
        optimizer=params.get("optimizer", base_tcfg.optimizer),
        weight_decay=params.get("weight_decay", base_tcfg.weight_decay),
        momentum=params.get("momentum", base_tcfg.momentum),
        aug_min_scale=params.get("aug_min_scale", base_tcfg.aug_min_scale),
    )
    return mcfg, tcfg


def run_search(
    scfg: SearchConfig, base_mcfg: ModelConfig, base_tcfg: TrainConfig, datasets, device
):
    trials = build_trials(scfg)

    train_ds, val_ds, test_ds = datasets
    s_train = _subset(train_ds, scfg.subset_size, base_tcfg.seed)
    s_val = _subset(
        val_ds,
        None if scfg.subset_size is None else max(scfg.subset_size // 4, 500),
        base_tcfg.seed + 1,
    )
    s_datasets = (s_train, s_val, test_ds)

    print(
        f"[search] method={scfg.method} | {len(trials)} trial(s) "
        f"| {scfg.search_epochs} epoch(s) each "
        f"| {len(s_train)} train / {len(s_val)} val images per trial"
    )

    results = []
    best_params, best_psnr = None, float("-inf")
    for i, params in enumerate(trials, 1):
        mcfg, tcfg = apply_params(params, base_mcfg, base_tcfg)
        print(f"[search] trial {i}/{len(trials)}: {params}")
        psnr = train_model(
            mcfg,
            tcfg,
            s_datasets,
            device,
            scfg.search_epochs,
            verbose=False,
            checkpoint=False,
        ).best_psnr
        print(f"[search]   -> val_psnr {psnr:.2f} dB")
        results.append((params, psnr))
        if psnr > best_psnr:
            best_params, best_psnr = params, psnr

    print(f"[search] best: {best_params} ({best_psnr:.2f} dB)")
    return best_params, best_psnr, results
