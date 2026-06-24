import glob
import os

import torch

import interpret as I
from colorspace import ABQuantizer
from config import ModelConfig, TrainConfig
from data import get_datasets
from model import build_model
from utils import get_device


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    state = ckpt.get("state_dict") or ckpt.get("model_state")
    model = build_model(mcfg).to(device)
    model.load_state_dict(state)
    model.eval()
    quant = (
        ABQuantizer(mcfg.grid_size, temperature=mcfg.decode_temperature, device=device)
        if mcfg.mode == "lab_classification"
        else None
    )
    history = ckpt.get("history")  # embedded by newer training runs (may be None)
    tcfg = (
        TrainConfig(**ckpt["train_config"]) if "train_config" in ckpt else TrainConfig()
    )
    return model, mcfg, quant, history, tcfg


def main():
    # config
    CKPT = "colorizer.pt"
    OUT_DIR = "analysis"
    LOG_GLOB = "logs/*"  # parsed for training curves / search results
    N_SAMPLES = 8  # images used in the per-image figures

    device = get_device()
    print(f"[device] {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    model, mcfg, quant, history, tcfg = load_model(CKPT, device)
    print(
        f"[model] mode={mcfg.mode} base_channels={mcfg.base_channels} depth={mcfg.depth} "
        f"data={tcfg.datasets} @ {tcfg.image_size}px"
    )

    _, _, test_ds = get_datasets(tcfg)
    batch = I.get_test_batch(test_ds, N_SAMPLES, device)

    # prefer history/search results embedded in the checkpoint, otherwise
    # parse the most recent training log
    trials = []
    if history is None:
        logs = sorted(glob.glob(LOG_GLOB))
        if logs:
            print(f"[log] parsing {logs[-1]}")
            history, trials = I.parse_training_log(logs[-1])

    bn_name, bn_layer = I.bottleneck_layer(model, device)
    convs = I.get_conv_layers(model)
    feat_layer = convs[len(convs) // 2][0]  # a mid-network layer for feature viz
    print(f"[layers] grad-cam target='{bn_name}', feature-viz layer='{feat_layer}'")

    def run(label, fn):
        try:
            path = fn()
            if path:
                print(f"  [ok] {label:24s} -> {path}")
            else:
                print(f"  [skip] {label:24s} (not applicable for mode={mcfg.mode})")
        except Exception as e:  # keep going even if one analysis fails
            print(f"  [skip] {label:24s} ({type(e).__name__}: {e})")

    print("[analysis] generating figures...")
    run(
        "training_curves",
        lambda: I.plot_training_curves(history, f"{OUT_DIR}/01_training_curves.png"),
    )
    run(
        "search_results",
        lambda: I.plot_search_results(trials, f"{OUT_DIR}/02_search_results.png"),
    )
    run(
        "first_layer_filters",
        lambda: I.plot_first_layer_filters(
            model, f"{OUT_DIR}/03_first_layer_filters.png"
        ),
    )
    run(
        "weight_distributions",
        lambda: I.plot_weight_distributions(
            model, f"{OUT_DIR}/04_weight_distributions.png"
        ),
    )
    run(
        "feature_maps",
        lambda: I.plot_feature_maps(
            model, batch, mcfg.mode, quant, device, f"{OUT_DIR}/05_feature_maps.png"
        ),
    )
    run(
        "feature_visualization",
        lambda: I.plot_feature_visualization(
            model,
            feat_layer,
            device,
            f"{OUT_DIR}/06_feature_visualization.png",
            colorize_mode=mcfg.mode,
            quant=quant,
        ),
    )
    run(
        "grad_cam",
        lambda: I.plot_grad_cam(
            model,
            batch,
            bn_layer,
            mcfg.mode,
            quant,
            device,
            f"{OUT_DIR}/07_grad_cam.png",
        ),
    )
    run(
        "saliency",
        lambda: I.plot_saliency(
            model, batch, mcfg.mode, quant, device, f"{OUT_DIR}/08_saliency.png"
        ),
    )
    run(
        "occlusion",
        lambda: I.plot_occlusion(
            model, batch, mcfg.mode, quant, device, f"{OUT_DIR}/09_occlusion.png"
        ),
    )
    run(
        "ab_distribution",
        lambda: I.plot_ab_distribution(
            model,
            test_ds,
            mcfg.mode,
            quant,
            device,
            f"{OUT_DIR}/10_ab_distribution.png",
        ),
    )
    run(
        "color_error_maps",
        lambda: I.plot_color_error_maps(
            model, batch, mcfg.mode, quant, device, f"{OUT_DIR}/11_color_error_maps.png"
        ),
    )
    run(
        "colorfulness_dist",
        lambda: I.plot_colorfulness_distribution(
            model,
            test_ds,
            mcfg.mode,
            quant,
            device,
            f"{OUT_DIR}/12_colorfulness_distribution.png",
        ),
    )
    run(
        "psnr_analysis",
        lambda: I.plot_psnr_analysis(
            model, test_ds, mcfg.mode, quant, device, f"{OUT_DIR}/13_psnr_analysis.png"
        ),
    )
    run(
        "temperature_sweep",
        lambda: I.plot_temperature_sweep(
            model,
            test_ds,
            mcfg.mode,
            quant,
            device,
            f"{OUT_DIR}/14_temperature_sweep.png",
        ),
    )
    print(f"[done] figures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
