import ast
import math
import os
import random
import re

import matplotlib

from model import UNet

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from colorspace import lab_to_rgb, rgb_to_lab
from engine import decode_to_rgb, prepare_batch


def get_conv_layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]


def module_by_name(model, name):
    return dict(model.named_modules())[name]


def conv_output_sizes(model, device):
    sizes = {}
    handles = []
    for name, m in get_conv_layers(model):
        handles.append(
            m.register_forward_hook(
                lambda mod, i, o, n=name: sizes.__setitem__(n, o.shape[-1])
            )
        )
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 32, 32, device=device))
    for h in handles:
        h.remove()
    return sizes


def bottleneck_layer(model, device):
    sizes = conv_output_sizes(model, device)
    name = min(sizes, key=lambda k: sizes[k])
    return name, module_by_name(model, name)


def colorfulness_scalar(pred, mode, quant):
    if mode == "lab_regression":
        ab = pred * 110.0
        chroma = (ab**2).sum(dim=1).clamp(min=1e-8).sqrt()
    elif mode == "lab_classification":
        ab = quant.soft_decode(pred)
        chroma = (ab**2).sum(dim=1).clamp(min=1e-8).sqrt()
    elif mode == "rgb_regression":
        chroma = pred.std(dim=1)
    else:
        raise ValueError(mode)
    return chroma.mean()


def colorfulness_metric(rgb):
    # Hasler-Susstrunk colourfulness per image
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    rg = rg.flatten(1)
    yb = yb.flatten(1)
    std = torch.sqrt(rg.std(dim=1) ** 2 + yb.std(dim=1) ** 2)
    mean = torch.sqrt(rg.mean(dim=1) ** 2 + yb.mean(dim=1) ** 2)
    return std + 0.3 * mean


def _to_img(t):
    # (3,H,W) or (H,W) tensor -> numpy for imshow
    t = t.detach().cpu()
    if t.dim() == 3:
        return t.permute(1, 2, 0).numpy()
    return t.numpy()


def get_test_batch(test_ds, n, device):
    imgs = torch.stack([test_ds[i][0] for i in range(n)])
    return imgs.to(device)


def parse_training_log(path):
    # Extract (history, trials) from a run's stdout log
    # @TODO: gotta remove cuz now we have history.json & embeded model history
    epoch_re = re.compile(
        r"epoch\s+(\d+)/\d+\s+\|\s+train_loss\s+([\d.]+)\s+\|\s+"
        r"val_loss\s+([\d.]+)\s+\|\s+val_psnr\s+([\d.]+)"
    )
    trial_re = re.compile(r"trial\s+\d+/\d+:\s+(\{.*\})")
    psnr_re = re.compile(r"->\s*val_psnr\s+([\d.]+)")

    history, trials, pending = [], [], None
    with open(path) as f:
        for line in f:
            m = epoch_re.search(line)
            if m:
                history.append(
                    {
                        "epoch": int(m.group(1)),
                        "train_loss": float(m.group(2)),
                        "val_loss": float(m.group(3)),
                        "val_psnr": float(m.group(4)),
                    }
                )
                continue
            m = trial_re.search(line)
            if m:
                try:
                    pending = ast.literal_eval(m.group(1))
                except (ValueError, SyntaxError):
                    pending = None
                continue
            m = psnr_re.search(line)
            if m and pending is not None:
                trials.append({**pending, "val_psnr": float(m.group(1))})
                pending = None
    return history, trials


def plot_training_curves(history, path):
    if not history:
        return None
    ep = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ep, [h["train_loss"] for h in history], label="train loss")
    ax1.plot(ep, [h["val_loss"] for h in history], label="val loss")
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss (log)")
    ax1.legend()
    ax1.set_title("Training / validation loss")

    ax2.plot(ep, [h["val_psnr"] for h in history], color="tab:green")
    best = max(history, key=lambda h: h["val_psnr"])
    ax2.axhline(best["val_psnr"], ls="--", c="gray", lw=1)
    ax2.scatter(
        [best["epoch"]],
        [best["val_psnr"]],
        color="red",
        zorder=5,
        label=f"best {best['val_psnr']:.2f} dB @ ep{best['epoch']}",
    )
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("val PSNR (dB)")
    ax2.legend()
    ax2.set_title("Validation PSNR")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_search_results(trials, path):
    if not trials:
        return None
    trials = sorted(trials, key=lambda t: t["val_psnr"])
    modes = sorted({t.get("mode", "?") for t in trials})
    cmap = {m: c for m, c in zip(modes, plt.cm.tab10.colors)}  # pyright: ignore[reportAttributeAccessIssue]
    colors = [cmap[t.get("mode", "?")] for t in trials]
    labels = [
        f"lr={t.get('lr')}, bs={t.get('batch_size')}, {t.get('optimizer')}, "
        f"ch={t.get('base_channels')}, d={t.get('depth')}"
        for t in trials
    ]

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(trials) + 1.5))
    ax.barh(range(len(trials)), [t["val_psnr"] for t in trials], color=colors)
    ax.set_yticks(range(len(trials)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("val PSNR (dB)")
    lo = min(t["val_psnr"] for t in trials)
    ax.set_xlim(lo - 0.3, max(t["val_psnr"] for t in trials) + 0.1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[m]) for m in modes]  # pyright: ignore[reportPrivateImportUsage]
    ax.legend(handles, modes, title="mode", fontsize=8)
    ax.set_title("Hyperparameter search trials (sorted)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_first_layer_filters(model, path):
    w = None
    for _, m in get_conv_layers(model):
        w = m.weight.detach().cpu()
        break
    if w is None:
        return None
    n = w.shape[0]
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.7, rows * 0.7))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            f = w[i, 0]
            ax.imshow(f, cmap="gray")
    fig.suptitle(f"First conv layer filters ({n} x {tuple(w.shape[2:])})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_weight_distributions(model, path):
    convs = get_conv_layers(model)
    data = [m.weight.detach().cpu().flatten().numpy() for _, m in convs]
    names = [n for n, _ in convs]
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(convs) + 2))
    ax.violinplot(data, vert=False, showextrema=False, widths=0.9)
    ax.set_yticks(range(1, len(names) + 1))
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(0, c="gray", lw=0.8)
    ax.set_xlabel("weight value")
    ax.set_title("Per-layer convolution weight distributions")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_feature_maps(
    model, rgb_image, mode, quant, device, path, layer_names=None, max_channels=16
):
    inp, _, _ = prepare_batch(rgb_image[:1], mode, quant)
    convs = get_conv_layers(model)
    if layer_names is None:  # spread a few layers across the network depth
        idxs = sorted(set([0, len(convs) // 3, 2 * len(convs) // 3, len(convs) - 1]))
        layer_names = [convs[i][0] for i in idxs]

    acts = {}
    handles = [
        module_by_name(model, n).register_forward_hook(
            lambda m, i, o, n=n: acts.__setitem__(n, o.detach())
        )
        for n in layer_names
    ]
    model.eval()
    with torch.no_grad():
        model(inp)
    for h in handles:
        h.remove()

    fig, axes = plt.subplots(
        len(layer_names),
        max_channels,
        figsize=(max_channels * 0.7, len(layer_names) * 0.8),
    )
    axes = np.array(axes).reshape(len(layer_names), max_channels)
    for r, name in enumerate(layer_names):
        a = acts[name][0]
        for c in range(max_channels):
            ax = axes[r, c]
            ax.axis("off")
            if c < a.shape[0]:
                ax.imshow(a[c].cpu(), cmap="viridis")
            if c == 0:
                ax.set_title(f"{name}\n{tuple(a.shape)}", fontsize=6, loc="left")
    fig.suptitle("Activation maps (first channels per layer)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _total_variation(x):
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (
        x[..., :, 1:] - x[..., :, :-1]
    ).abs().mean()


def activation_maximization(
    model,
    layer,
    channel,
    device,
    steps=160,
    lr=0.08,
    tv_weight=0.05,
    l2_weight=1e-3,
    size=32,
):
    model.eval()
    param = (torch.randn(1, 1, size, size, device=device) * 0.1).requires_grad_(True)
    opt = torch.optim.AdamW([param], lr=lr)
    store = {}
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__("o", o))
    for _ in range(steps):
        opt.zero_grad()
        dy, dx = random.randint(-2, 2), random.randint(-2, 2)  # jitter = robustness
        x = torch.tanh(torch.roll(param, shifts=(dy, dx), dims=(2, 3)))
        model(x)
        act = store["o"][0, channel].mean()
        loss = -act + tv_weight * _total_variation(x) + l2_weight * (x**2).mean()
        loss.backward()
        opt.step()
    h.remove()
    return torch.tanh(param).detach()


def plot_feature_visualization(
    model, layer_name, device, path, n_channels=8, colorize_mode=None, quant=None
):
    layer = module_by_name(model, layer_name)
    n_out = layer.weight.shape[0]
    chans = list(range(min(n_channels, n_out)))

    fig, axes = plt.subplots(2, len(chans), figsize=(len(chans) * 1.3, 3.0))
    axes = np.array(axes).reshape(2, len(chans))
    for j, ch in enumerate(chans):
        opt_in = activation_maximization(model, layer, ch, device)
        gray = ((opt_in + 1) / 2).clamp(0, 1)
        axes[0, j].imshow(gray[0, 0].cpu(), cmap="gray")
        axes[0, j].axis("off")
        axes[0, j].set_title(f"ch {ch}", fontsize=7)
        with torch.no_grad():  # how the model itself would colour that pattern
            pred = model(opt_in)
            L = (opt_in + 1) * 50.0
            rgb = decode_to_rgb(pred, L, colorize_mode, quant)
        axes[1, j].imshow(_to_img(rgb[0]))
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=8)
    axes[1, 0].set_ylabel("colorized", fontsize=8)
    fig.suptitle(f"Feature visualization (activation max) — layer '{layer_name}'")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def grad_cam(model: UNet, rgb_image, target_layer, mode, quant, device):
    model.eval()
    inp, _, L = prepare_batch(rgb_image[:1], mode, quant)
    acts, grads = {}, {}

    # jesus take the wheel
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = target_layer.register_full_backward_hook(
        lambda m, gi, go: grads.__setitem__("g", go[0])
    )

    pred = model(inp)
    model.zero_grad()
    colorfulness_scalar(pred, mode, quant).backward()
    h1.remove()
    h2.remove()

    A, G = acts["a"], grads["g"]  # (1,C,h,w)
    weights = G.mean(dim=(2, 3), keepdim=True)  # global-avg-pooled grads
    cam = (weights * A).sum(dim=1, keepdim=True).relu()
    cam = F.interpolate(cam, size=inp.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu(), L.detach().cpu()


def plot_grad_cam(model, rgb_image, target_layer, mode, quant, device, path, n=6):
    fig, axes = plt.subplots(3, n, figsize=(n * 1.4, 4.4))
    for c in range(n):
        cam, L = grad_cam(
            model, rgb_image[c : c + 1], target_layer, mode, quant, device
        )
        gray = (L[0, 0] / 100.0).clamp(0, 1)
        axes[0, c].imshow(gray, cmap="gray")
        axes[0, c].axis("off")
        axes[1, c].imshow(cam, cmap="jet")
        axes[1, c].axis("off")
        axes[2, c].imshow(gray, cmap="gray")
        axes[2, c].imshow(cam, cmap="jet", alpha=0.5)
        axes[2, c].axis("off")
    for r, t in enumerate(["input (gray)", "Grad-CAM", "overlay"]):
        axes[r, 0].set_ylabel(t, fontsize=8)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
    fig.suptitle("Grad-CAM — pixels driving the model to add colour")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_saliency(model, rgb_image, mode, quant, device, path, n=6):
    model.eval()
    fig, axes = plt.subplots(2, n, figsize=(n * 1.4, 3.0))
    for c in range(n):
        inp, _, L = prepare_batch(rgb_image[c : c + 1], mode, quant)
        inp = inp.clone().requires_grad_(True)
        pred = model(inp)
        model.zero_grad()
        colorfulness_scalar(pred, mode, quant).backward()
        assert inp.grad, "grad is None?"
        sal = inp.grad[0, 0].abs().cpu()
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        axes[0, c].imshow((L[0, 0] / 100).clamp(0, 1).cpu(), cmap="gray")
        axes[0, c].axis("off")
        axes[1, c].imshow(sal, cmap="hot")
        axes[1, c].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=8)
    axes[0, 0].axis("on")
    axes[1, 0].set_ylabel("saliency", fontsize=8)
    axes[1, 0].axis("on")
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Saliency — |d colourfulness / d input pixel|")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


@torch.no_grad()
def occlusion_sensitivity(model, rgb_image, mode, quant, device, patch=8, stride=2):
    model.eval()
    inp, _, L = prepare_batch(rgb_image[:1], mode, quant)
    base_rgb = decode_to_rgb(model(inp), L, mode, quant)
    H, W = inp.shape[-2:]
    heat = torch.zeros(H, W)
    count = torch.zeros(H, W)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2, x2 = min(y + patch, H), min(x + patch, W)
            occ = inp.clone()
            occ[:, :, y:y2, x:x2] = 0.0  # 0 == mid-grey (L=50) after de-norm
            rgb = decode_to_rgb(model(occ), L, mode, quant)
            delta = (rgb - base_rgb).abs().mean().item()
            heat[y:y2, x:x2] += delta
            count[y:y2, x:x2] += 1
    return (heat / count.clamp(min=1)).cpu(), L[0, 0].cpu()


def plot_occlusion(model, rgb_image, mode, quant, device, path, n=4):
    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.2))
    for c in range(n):
        heat, L = occlusion_sensitivity(
            model, rgb_image[c : c + 1], mode, quant, device
        )
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        axes[0, c].imshow((L / 100).clamp(0, 1), cmap="gray")
        axes[0, c].axis("off")
        axes[1, c].imshow(heat, cmap="inferno")
        axes[1, c].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=8)
    axes[0, 0].axis("on")
    axes[1, 0].set_ylabel("sensitivity", fontsize=8)
    axes[1, 0].axis("on")
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Occlusion sensitivity — how much each region changes the colours")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


@torch.no_grad()
def plot_ab_distribution(model, test_ds, mode, quant, device, path, n_images=300):
    model.eval()
    rgb = get_test_batch(test_ds, n_images, device)
    inp, _, L = prepare_batch(rgb, mode, quant)
    pred_rgb = decode_to_rgb(model(inp), L, mode, quant)

    gt_ab = rgb_to_lab(rgb)[:, 1:3].flatten(2).reshape(2, -1).cpu().numpy()
    pr_ab = (
        rgb_to_lab(pred_rgb)[:, 1:3].permute(1, 0, 2, 3).reshape(2, -1).cpu().numpy()
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    rng = [[-90, 90], [-90, 90]]
    for ax, (a, b), title in [
        (axes[0], gt_ab, "ground truth"),
        (axes[1], pr_ab, "predicted"),
    ]:
        ax.hist2d(
            a,
            b,
            bins=80,
            range=rng,
            cmap="magma",
            norm=matplotlib.colors.LogNorm(),  # pyright: ignore[reportAttributeAccessIssue]
        )
        ax.axhline(0, c="w", lw=0.5)
        ax.axvline(0, c="w", lw=0.5)
        ax.set_title(f"ab distribution — {title}")
        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.set_aspect("equal")
    fig.suptitle("Colour distribution in the ab plane (note predicted desaturation)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


@torch.no_grad()
def plot_color_error_maps(model, rgb_image, mode, quant, device, path, n=6):
    model.eval()
    inp, _, L = prepare_batch(rgb_image[:n], mode, quant)
    pred_rgb = decode_to_rgb(model(inp), L, mode, quant)
    err = (pred_rgb - rgb_image[:n]).abs().mean(dim=1)  # per-pixel RGB error

    rows = [
        ("input", None),
        ("colorized", pred_rgb),
        ("ground truth", rgb_image[:n]),
        ("abs error", err),
    ]
    fig, axes = plt.subplots(4, n, figsize=(n * 1.4, 5.6))
    for c in range(n):
        gray = (L[c, 0] / 100).clamp(0, 1).cpu()
        axes[0, c].imshow(gray, cmap="gray")
        axes[1, c].imshow(_to_img(pred_rgb[c]))
        axes[2, c].imshow(_to_img(rgb_image[c]))
        im = axes[3, c].imshow(err[c].cpu(), cmap="hot", vmin=0, vmax=float(err.max()))
        for r in range(4):
            axes[r, c].axis("off")
    for r, (t, _) in enumerate(rows):
        axes[r, 0].set_ylabel(t, fontsize=8)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
    fig.colorbar(im, ax=axes[3, :].tolist(), fraction=0.02)  # pyright: ignore[reportPossiblyUnboundVariable]
    fig.suptitle("Per-pixel colour reconstruction error")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


@torch.no_grad()
def plot_colorfulness_distribution(
    model, test_ds, mode, quant, device, path, n_images=500
):
    model.eval()
    rgb = get_test_batch(test_ds, n_images, device)
    inp, _, L = prepare_batch(rgb, mode, quant)
    pred_rgb = decode_to_rgb(model(inp), L, mode, quant)
    gt_c = colorfulness_metric(rgb).cpu().numpy()
    pr_c = colorfulness_metric(pred_rgb).cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(gt_c.max(), pr_c.max()), 40)
    ax.hist(gt_c, bins=bins, alpha=0.6, label=f"ground truth (mean {gt_c.mean():.3f})")
    ax.hist(pr_c, bins=bins, alpha=0.6, label=f"predicted (mean {pr_c.mean():.3f})")
    ax.set_xlabel("Hasler-Susstrunk colourfulness")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title("Colourfulness: predicted vs ground truth")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


@torch.no_grad()
def plot_temperature_sweep(
    model, test_ds, mode, quant, device, path, temps=(0.1, 0.2, 0.38, 0.6, 1.0), n=6
):
    if mode != "lab_classification" or quant is None:
        return None
    model.eval()
    rgb = get_test_batch(test_ds, n, device)
    inp, _, L = prepare_batch(rgb, mode, quant)
    logits = model(inp)

    rows = len(temps) + 2  # gray input, one row per temperature, ground truth
    fig, axes = plt.subplots(rows, n, figsize=(n * 1.4, rows * 1.0))
    for c in range(n):
        axes[0, c].imshow((L[c, 0] / 100).clamp(0, 1).cpu(), cmap="gray")
    for ti, t in enumerate(temps):
        ab = quant.soft_decode(logits, temperature=t)
        rgb_t = lab_to_rgb(torch.cat([L, ab], dim=1))
        cfn = colorfulness_metric(rgb_t).mean().item()
        for c in range(n):
            axes[ti + 1, c].imshow(_to_img(rgb_t[c]))
        axes[ti + 1, 0].set_ylabel(f"T={t}\ncf {cfn:.2f}", fontsize=7)
    for c in range(n):
        axes[-1, c].imshow(_to_img(rgb[c]))
    axes[0, 0].set_ylabel("input", fontsize=7)
    axes[-1, 0].set_ylabel("truth", fontsize=7)
    for r in range(rows):
        for c in range(n):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if c != 0:
                axes[r, c].axis("off")
    fig.suptitle("Decode-temperature sweep — lower T = more vivid (cf = colourfulness)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


@torch.no_grad()
def plot_psnr_analysis(model, test_ds, mode, quant, device, path, n_images=2000, k=6):
    model.eval()
    rgb = get_test_batch(test_ds, n_images, device)
    psnrs = []
    preds = []
    bs = 256
    for s in range(0, len(rgb), bs):
        chunk = rgb[s : s + bs]
        inp, _, L = prepare_batch(chunk, mode, quant)
        pr = decode_to_rgb(model(inp), L, mode, quant)
        mse = ((pr - chunk) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
        psnrs.append((10 * torch.log10(1 / mse)).cpu())
        preds.append(pr.cpu())
    psnrs = torch.cat(psnrs)
    preds = torch.cat(preds)
    order = torch.argsort(psnrs)
    worst, best = order[:k], order[-k:].flip(0)

    fig = plt.figure(figsize=(max(2 * k, 8), 6.5))
    gs = fig.add_gridspec(3, 2 * k, height_ratios=[2.2, 1, 1], hspace=0.35, top=0.9)
    axh = fig.add_subplot(gs[0, :])
    axh.hist(psnrs.numpy(), bins=50, color="tab:blue")
    axh.axvline(
        float(psnrs.mean()), c="red", ls="--", label=f"mean {psnrs.mean():.2f} dB"
    )
    axh.set_xlabel("per-image PSNR (dB)")
    axh.set_ylabel("count")
    axh.legend()
    axh.set_title(f"PSNR distribution over {len(psnrs)} test images")

    for j, idx in enumerate(best):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(_to_img(preds[idx]))
        ax.axis("off")
        ax.set_title(f"{psnrs[idx]:.1f}", fontsize=7)
        ax = fig.add_subplot(gs[2, j])
        ax.imshow(_to_img(rgb[idx].cpu()))
        ax.axis("off")
    for j, idx in enumerate(worst):
        ax = fig.add_subplot(gs[1, k + j])
        ax.imshow(_to_img(preds[idx]))
        ax.axis("off")
        ax.set_title(f"{psnrs[idx]:.1f}", fontsize=7)
        ax = fig.add_subplot(gs[2, k + j])
        ax.imshow(_to_img(rgb[idx].cpu()))
        ax.axis("off")
    # group headers (centred over each half) and row labels on the left
    pos = gs[1, :].get_position(fig)
    fig.text(0.25, pos.y1 + 0.015, f"BEST {k}", fontsize=10, ha="center", weight="bold")
    fig.text(
        0.75, pos.y1 + 0.015, f"WORST {k}", fontsize=10, ha="center", weight="bold"
    )
    fig.text(
        0.085,
        (gs[1, 0].get_position(fig).y0 + gs[1, 0].get_position(fig).y1) / 2,
        "predicted",
        fontsize=8,
        va="center",
        ha="right",
    )
    fig.text(
        0.085,
        (gs[2, 0].get_position(fig).y0 + gs[2, 0].get_position(fig).y1) / 2,
        "truth",
        fontsize=8,
        va="center",
        ha="right",
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
