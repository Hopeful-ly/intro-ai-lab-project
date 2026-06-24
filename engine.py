import json
import math
import os
from dataclasses import asdict, dataclass
from typing import TypedDict

import torch
import torch.nn.functional as F

from colorspace import ABQuantizer, lab_to_rgb, rgb_to_lab
from config import ModelConfig, TrainConfig
from data import make_loader
from model import UNet, build_model
from utils import set_seed


class HistoryEntry(TypedDict):
    epoch: int
    train_loss: float
    val_loss: float
    val_psnr: float


class BestEntry(TypedDict):
    epoch: int
    val_loss: float
    val_psnr: float


@dataclass
class TrainResult:
    model: UNet  # the trained model (left at its final-epoch weights)
    best_psnr: float  # best validation PSNR seen during training
    quant: ABQuantizer | None  # ABQuantizer (classification) or None
    history: list[HistoryEntry]  # per-epoch {epoch, train_loss, val_loss, val_psnr}
    best_state: dict  # CPU copy of the state_dict at the best-PSNR epoch


def prepare_batch(rgb: torch.Tensor, mode: str, quant: ABQuantizer | None):
    lab = rgb_to_lab(rgb)
    L = lab[:, 0:1]  # [0, 100]
    ab = lab[:, 1:3]  # ~[-110, 110]
    inp = L / 50.0 - 1.0  # -> ~[-1, 1]

    if mode == "lab_regression":
        target = ab / 110.0
    elif mode == "rgb_regression":
        target = rgb
    elif mode == "lab_classification":
        assert quant is not None, "quant must be provided for lab_classification mode"
        target = quant.ab_to_index(ab)
    else:
        raise ValueError(mode)
    return inp, target, L


def compute_loss(pred, target, mode: str, weight=None) -> torch.Tensor:
    if mode == "lab_classification":
        return F.cross_entropy(pred, target, weight=weight)
    return F.mse_loss(pred, target)


@torch.no_grad()
def compute_class_weights(
    quant: ABQuantizer,
    train_ds,
    device,
    lam: float = 0.5,
    n_images: int = 5000,
    batch_size: int = 256,
):
    counts = torch.zeros(quant.num_classes, device=device)
    n = min(n_images, len(train_ds))
    for s in range(0, n, batch_size):
        idxs = range(s, min(s + batch_size, n))
        rgb = torch.stack([train_ds[i][0] for i in idxs]).to(device)
        idx = quant.ab_to_index(rgb_to_lab(rgb)[:, 1:3]).flatten()
        counts += torch.bincount(idx, minlength=quant.num_classes).float()

    prior = counts / counts.sum().clamp(min=1)
    w = 1.0 / ((1 - lam) * prior + lam / quant.num_classes)
    w = w / (prior * w).sum().clamp(min=1e-8)  # E_p[w] = 1
    w[prior == 0] = 0.0  # bins never seen contribute nothing
    return w


def decode_to_rgb(pred, L, mode: str | None, quant: ABQuantizer | None) -> torch.Tensor:
    if mode == "lab_regression":
        rgb = lab_to_rgb(torch.cat([L, pred * 110.0], dim=1))
    elif mode == "rgb_regression":
        rgb = pred.clamp(0, 1)
    elif mode == "lab_classification":
        assert quant is not None, "quant must be provided for lab_classification mode"
        ab = quant.soft_decode(pred)
        rgb = lab_to_rgb(torch.cat([L, ab], dim=1))
    else:
        raise ValueError(mode)
    return rgb


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_psnr,
    mcfg: ModelConfig,
    tcfg: TrainConfig,
    history=None,
):
    torch.save(
        {
            "epoch": epoch,  # number of epochs completed
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_psnr": best_psnr,
            "model_config": asdict(mcfg),
            "train_config": asdict(tcfg),
            "history": history
            if history is not None
            else [],  # curve so far (for resume + plots)
        },
        path,
    )


def build_optimizer(tcfg: TrainConfig, model):
    if tcfg.optimizer == "adam":
        return torch.optim.AdamW(
            model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
        )
    if tcfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=tcfg.lr,
            momentum=tcfg.momentum,
            weight_decay=tcfg.weight_decay,
        )
    raise ValueError(f"unknown optimizer: {tcfg.optimizer}")


def train_one_epoch(
    model, loader, optimizer, scaler, mode, quant, device, use_amp, loss_weight=None
):
    model.train()
    running, n = 0.0, 0
    for rgb, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        inp, target, _ = prepare_batch(rgb, mode, quant)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(inp)
            loss = compute_loss(pred, target, mode, weight=loss_weight)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item() * rgb.size(0)
        n += rgb.size(0)
    return running / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, mode, quant, device, loss_weight=None):
    model.eval()
    total_mse, total_loss, n = 0.0, 0.0, 0
    for rgb, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        inp, target, L = prepare_batch(rgb, mode, quant)
        pred = model(inp)
        total_loss += compute_loss(
            pred, target, mode, weight=loss_weight
        ).item() * rgb.size(0)

        rgb_pred = decode_to_rgb(pred, L, mode, quant)
        per_img_mse = ((rgb_pred - rgb) ** 2).mean(dim=[1, 2, 3])
        total_mse += per_img_mse.sum().item()
        n += rgb.size(0)

    mean_mse = total_mse / max(n, 1)
    psnr = 10 * math.log10(1.0 / max(mean_mse, 1e-10))
    return psnr, total_loss / max(n, 1)


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_model(
    mcfg: ModelConfig,
    tcfg: TrainConfig,
    datasets,
    device,
    epochs: int,
    verbose: bool = True,
    checkpoint: bool = True,
) -> TrainResult:
    set_seed(tcfg.seed)
    train_ds, val_ds, _ = datasets

    quant, loss_weight = None, None
    if mcfg.mode == "lab_classification":
        quant = ABQuantizer(
            mcfg.grid_size, temperature=mcfg.decode_temperature, device=device
        )
        if mcfg.class_rebalance:
            loss_weight = compute_class_weights(
                quant, train_ds, device, mcfg.rebalance_lambda
            )
            if verbose:
                print(
                    f"  [rebalance] colour-bin weights: "
                    f"min {loss_weight.min():.2f}  max {loss_weight.max():.2f}"
                )

    model = build_model(mcfg).to(device)
    optimizer = build_optimizer(tcfg, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )

    use_amp = device.type == "cuda" and tcfg.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # pyright: ignore[reportPrivateImportUsage]

    start_epoch, best_psnr, resumed_history = 0, float("-inf"), []
    if tcfg.resume_from:
        ckpt = torch.load(tcfg.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch, best_psnr = ckpt["epoch"], ckpt["best_psnr"]

        resumed_history = list(ckpt.get("history") or [])
        print(
            f"  [resume] continuing from {tcfg.resume_from} at epoch {start_epoch} "
            f"({len(resumed_history)} prior epoch(s) of history)"
        )

    do_ckpt = checkpoint and tcfg.checkpoint_every > 0
    if do_ckpt:
        os.makedirs(tcfg.checkpoint_dir, exist_ok=True)

    train_loader = make_loader(
        train_ds, tcfg.batch_size, True, device, tcfg.num_workers
    )
    val_loader = make_loader(val_ds, tcfg.batch_size, False, device, tcfg.num_workers)

    best_state = _cpu_state(model)  # best-validation state
    history = resumed_history  # continue the curve across a resume
    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            mcfg.mode,
            quant,
            device,
            use_amp,
            loss_weight,
        )
        val_psnr, val_loss = evaluate(
            model, val_loader, mcfg.mode, quant, device, loss_weight
        )
        scheduler.step()
        improved = val_psnr > best_psnr
        if improved:
            best_psnr = val_psnr
            best_state = _cpu_state(model)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
            }
        )
        if verbose:
            print(
                f"  epoch {epoch + 1:3d}/{epochs} | train_loss {train_loss:.4f} "
                f"| val_loss {val_loss:.4f} | val_psnr {val_psnr:.2f} dB"
            )

        if do_ckpt:
            args = (
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                best_psnr,
                mcfg,
                tcfg,
                history,
            )
            is_last = (epoch + 1) == epochs
            if (epoch + 1) % tcfg.checkpoint_every == 0 or is_last:
                path = os.path.join(tcfg.checkpoint_dir, f"ckpt_epoch{epoch + 1}.pt")
                save_checkpoint(path, *args)
                save_checkpoint(
                    os.path.join(tcfg.checkpoint_dir, "ckpt_last.pt"), *args
                )
                if verbose:
                    print(f"    [checkpoint] saved {path}")
            if improved:
                save_checkpoint(
                    os.path.join(tcfg.checkpoint_dir, "ckpt_best.pt"), *args
                )

            # persist the curve every epoch so a mid-run stop (ctrl+c / crash) keeps history.
            # if only i did this sooner...
            hist_path = os.path.join(tcfg.checkpoint_dir, "history.json")
            tmp_path = hist_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(history, f, indent=2)
            os.replace(tmp_path, hist_path)

    return TrainResult(model, best_psnr, quant, history, best_state)
