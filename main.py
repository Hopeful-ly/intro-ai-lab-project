from config import ModelConfig, SearchConfig, TrainConfig
from data import get_datasets, make_loader
from engine import evaluate, train_model
from search import apply_params, run_search
from utils import get_device
from visualize import save_model, save_samples


def main():
    # -- CONFIG --
    RUN_SEARCH = True  # False skips the search and trains the base config directly

    base_model = ModelConfig()
    base_train = TrainConfig(epochs=40)
    search = SearchConfig(method="random", n_trials=50, search_epochs=8)
    # -- CONFIG --

    device = get_device()
    print(f"[device] using: {device}")

    datasets = get_datasets(base_train)
    train_ds, val_ds, test_ds = datasets
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    model_cfg, train_cfg = base_model, base_train
    if RUN_SEARCH:
        best_params, _, _ = run_search(search, base_model, base_train, datasets, device)
        model_cfg, train_cfg = apply_params(best_params, base_model, base_train)  # pyright: ignore[reportArgumentType]
        print(
            f"[main] retraining best config for {train_cfg.epochs} epochs: "
            f"mode={model_cfg.mode}, base_channels={model_cfg.base_channels}, "
            f"depth={model_cfg.depth}, lr={train_cfg.lr}, bs={train_cfg.batch_size}, "
            f"opt={train_cfg.optimizer}, wd={train_cfg.weight_decay}"
        )

    result = train_model(
        model_cfg, train_cfg, datasets, device, train_cfg.epochs, verbose=True
    )
    model, quant, history = result.model, result.quant, result.history
    print(f"[main] best validation PSNR: {result.best_psnr:.2f} dB")

    test_loader = make_loader(test_ds, train_cfg.batch_size, False, device)
    test_psnr, _ = evaluate(model, test_loader, model_cfg.mode, quant, device)
    print(f"[main] final-epoch test PSNR: {test_psnr:.2f} dB")

    save_model(model, model_cfg, "colorizer.pt", history=history, train_cfg=train_cfg)
    model.load_state_dict(result.best_state)
    best_test_psnr, _ = evaluate(model, test_loader, model_cfg.mode, quant, device)
    print(f"[main] best-epoch  test PSNR: {best_test_psnr:.2f} dB")
    save_model(model, model_cfg, "best.pt", history=history, train_cfg=train_cfg)
    save_samples(
        model, test_ds, model_cfg, quant, device, "samples.png"
    )  # uses best weights


if __name__ == "__main__":
    main()
