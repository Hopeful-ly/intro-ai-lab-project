import math

import numpy as np
import torch
import torch.nn.functional as F

from colorspace import rgb_to_lab
from data import get_datasets
from engine import decode_to_rgb, prepare_batch

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _gaussian_window(size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    w2d = g[:, None] * g[None, :]
    return w2d.expand(channels, 1, size, size).contiguous()


def ssim_per_image(x, y, window, size=11):
    # Mean SSIM per image for (B,3,H,W) inputs in [0,1] -> (B,)
    c = x.shape[1]
    pad = size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=c)
    mu_y = F.conv2d(y, window, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sig_x = F.conv2d(x * x, window, padding=pad, groups=c) - mu_x2
    sig_y = F.conv2d(y * y, window, padding=pad, groups=c) - mu_y2
    sig_xy = F.conv2d(x * y, window, padding=pad, groups=c) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    smap = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2)
    )
    return smap.mean(dim=(1, 2, 3))


def colorfulness_metric(rgb):
    # Hasler-Susstrunk colourfulness per image. (B,3,H,W) in [0,1] -> (B,)
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    rg = (R - G).flatten(1)
    yb = (0.5 * (R + G) - B).flatten(1)
    std = torch.sqrt(rg.std(dim=1) ** 2 + yb.std(dim=1) ** 2)
    mean = torch.sqrt(rg.mean(dim=1) ** 2 + yb.mean(dim=1) ** 2)
    return std + 0.3 * mean


AB_THRESHOLDS = [2, 5, 10, 15, 25]  # euclidean ab distance


@torch.no_grad()
def evaluate_scores(model, dataset, mode, quant, device, n_images=None, batch_size=256):
    # compute a dictionary of colorization metrics over dataset
    model.eval()
    n = len(dataset) if n_images is None else min(n_images, len(dataset))
    window = _gaussian_window(channels=3, device=device)

    psnrs, ssims, labels = [], [], []
    sum_mse = 0.0
    sum_rgb_mae = sum_ab_mae = 0.0
    sum_cf_pred = sum_cf_gt = 0.0
    thr_hits = {t: 0.0 for t in AB_THRESHOLDS}
    top1 = top5 = pix_total = 0.0
    n_seen = 0

    for s in range(0, n, batch_size):
        idxs = range(s, min(s + batch_size, n))
        rgb = torch.stack([dataset[i][0] for i in idxs]).to(device)
        labels.extend(int(dataset[i][1]) for i in idxs)
        inp, _, L = prepare_batch(rgb, mode, quant)
        logits = model(inp)
        pred_rgb = decode_to_rgb(logits, L, mode, quant)

        mse = ((pred_rgb - rgb) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
        sum_mse += mse.sum().item()
        psnrs.append((10 * torch.log10(1.0 / mse)).cpu())
        ssims.append(ssim_per_image(pred_rgb, rgb, window).cpu())
        sum_rgb_mae += (pred_rgb - rgb).abs().mean().item() * len(idxs)

        gt_ab = rgb_to_lab(rgb)[:, 1:3]
        pred_ab = rgb_to_lab(pred_rgb)[:, 1:3]
        sum_ab_mae += (pred_ab - gt_ab).abs().mean().item() * len(idxs)
        dist = ((pred_ab - gt_ab) ** 2).sum(dim=1).clamp(min=0).sqrt()  # (B,H,W)
        for t in AB_THRESHOLDS:
            thr_hits[t] += (dist <= t).float().mean().item() * len(idxs)

        sum_cf_pred += colorfulness_metric(pred_rgb).sum().item()
        sum_cf_gt += colorfulness_metric(rgb).sum().item()

        if mode == "lab_classification":
            gt_idx = quant.ab_to_index(gt_ab)  # (B,H,W)
            top1 += (logits.argmax(dim=1) == gt_idx).float().sum().item()
            t5 = logits.topk(5, dim=1).indices  # (B,5,H,W)
            top5 += (t5 == gt_idx.unsqueeze(1)).any(dim=1).float().sum().item()
            pix_total += gt_idx.numel()

        n_seen += len(idxs)

    psnrs = torch.cat(psnrs).numpy()
    ssims = torch.cat(ssims).numpy()
    labels = np.array(labels)

    scores = {
        "n_images": n_seen,
        "mode": mode,
        "psnr": {
            "mean": float(psnrs.mean()),
            "median": float(np.median(psnrs)),
            "std": float(psnrs.std()),
            "min": float(psnrs.min()),
            "max": float(psnrs.max()),
            "p05": float(np.percentile(psnrs, 5)),
            "p95": float(np.percentile(psnrs, 95)),
            # psnr from the dataset-mean mse -- matches the training/search metric
            # (differs from the per-image mean above by jensen's inequality)
            "dataset": 10 * math.log10(1.0 / max(sum_mse / n_seen, 1e-10)),
        },
        "ssim": float(ssims.mean()),
        "rgb_mae": sum_rgb_mae / n_seen,
        "ab_mae": sum_ab_mae / n_seen,
        "colorfulness": {
            "pred": sum_cf_pred / n_seen,
            "gt": sum_cf_gt / n_seen,
            "ratio": (sum_cf_pred / n_seen) / (sum_cf_gt / n_seen + 1e-8),
        },
        "ab_accuracy": {t: thr_hits[t] / n_seen for t in AB_THRESHOLDS},
        "ab_auc": float(np.mean([thr_hits[t] / n_seen for t in AB_THRESHOLDS])),
        "per_class_psnr": {
            CIFAR10_CLASSES[c]: float(psnrs[labels == c].mean())
            for c in range(10)
            if (labels == c).any()
        },
    }
    if mode == "lab_classification" and pix_total > 0:
        scores["pixel_topk"] = {"top1": top1 / pix_total, "top5": top5 / pix_total}
    return scores


def print_report(scores):
    p = scores["psnr"]
    cf = scores["colorfulness"]
    line = "=" * 58
    print(line)
    print(f" Colorization scores  ({scores['n_images']} images, mode={scores['mode']})")
    print(line)
    print(" Reconstruction quality")
    print(
        f"   PSNR (RGB)      mean {p['mean']:6.2f} dB   median {p['median']:6.2f}   "
        f"std {p['std']:.2f}"
    )
    print(
        f"                   range [{p['min']:.1f}, {p['max']:.1f}]   "
        f"p05 {p['p05']:.1f}   p95 {p['p95']:.1f}"
    )
    print(
        f"   PSNR (dataset)  {p['dataset']:6.2f} dB   (from mean MSE; matches "
        f"the training metric)"
    )
    print(f"   SSIM            {scores['ssim']:.4f}")
    print(f"   RGB MAE         {scores['rgb_mae']:.4f}    (per channel, [0,1])")
    print(f"   ab MAE          {scores['ab_mae']:.3f}     (CIELab colour error)")
    print(" Colour fidelity")
    print(
        f"   Colourfulness   pred {cf['pred']:.3f}   gt {cf['gt']:.3f}   "
        f"ratio {cf['ratio']:.2f}  (1.0 = matches truth)"
    )
    acc = scores["ab_accuracy"]
    acc_str = "   ".join(f"<={t}: {acc[t] * 100:4.1f}%" for t in AB_THRESHOLDS)
    print(f"   ab pixel acc    {acc_str}")
    print(
        f"   ab AuC          {scores['ab_auc'] * 100:.1f}%   "
        f"(mean over thresholds — higher = more accurate colour)"
    )
    if "pixel_topk" in scores:
        tk = scores["pixel_topk"]
        print(
            f"   colour-bin acc  top-1 {tk['top1'] * 100:.1f}%   top-5 {tk['top5'] * 100:.1f}%"
        )
    print(" Per-class PSNR (dB)")
    items = sorted(scores["per_class_psnr"].items(), key=lambda kv: kv[1])
    for i in range(0, len(items), 2):
        row = items[i : i + 2]
        print("   " + "   ".join(f"{k:11s} {v:5.2f}" for k, v in row))
    print(line)


def main():
    # config (edit here)
    CKPT = "colorizer.pt"
    N_IMAGES = None  # None to full test (10k) or int to subsample
    from analyze import load_model
    from utils import get_device

    device = get_device()
    print(f"[device] {device}")
    model, mcfg, quant, _, tcfg = load_model(CKPT, device)
    _, _, test_ds = get_datasets(tcfg)  # same data the model was trained on
    scores = evaluate_scores(
        model, test_ds, mcfg.mode, quant, device, n_images=N_IMAGES
    )
    print_report(scores)


if __name__ == "__main__":
    main()
