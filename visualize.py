import matplotlib

from model import UNet

matplotlib.use("Agg")  # headless-safe backend
from dataclasses import asdict

import matplotlib.pyplot as plt
import torch

from data import make_loader
from engine import decode_to_rgb, prepare_batch


def save_model(
    model, mcfg, path: str = "colorizer.pt", history=None, train_cfg=None
) -> None:
    payload = {"state_dict": model.state_dict(), "model_config": asdict(mcfg)}
    if history is not None:
        payload["history"] = history
    if train_cfg is not None:
        payload["train_config"] = asdict(train_cfg)
    torch.save(payload, path)
    print(f"[save] model -> {path}")


@torch.no_grad()
def save_samples(
    model: UNet, test_ds, mcfg, quant, device, path: str = "samples.png", n: int = 8
) -> None:
    model.eval()
    loader = make_loader(test_ds, batch_size=n, shuffle=False, device=device)
    rgb, _ = next(iter(loader))
    rgb = rgb.to(device)

    inp, _, L = prepare_batch(rgb, mcfg.mode, quant)
    pred = model(inp)
    rgb_pred = decode_to_rgb(pred, L, mcfg.mode, quant)

    gray = (L / 100.0).clamp(0, 1).cpu()
    rgb_pred = rgb_pred.cpu()
    rgb = rgb.cpu()

    rows = [
        ("input (gray)", gray, True),
        ("colorized", rgb_pred, False),
        ("ground truth", rgb, False),
    ]
    fig, axes = plt.subplots(3, n, figsize=(1.4 * n, 4.6))
    for r, (title, batch, is_gray) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            img = batch[c, 0] if is_gray else batch[c].permute(1, 2, 0)
            ax.imshow(img, cmap="gray" if is_gray else None, vmin=0, vmax=1)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(title, rotation=90, size="medium")
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle(f"UNet colorization ({mcfg.mode})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[save] samples -> {path}")
