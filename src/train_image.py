"""Train MobileNetV2 on existing Train/Validation CSVs. Never reads Test."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from image_baseline.common import (
    ImageTable, load_labels, check_disjoint, build_model, run_epoch,
    save_json, seed_everything, choose_device, digest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/image_text"))
    parser.add_argument("--image-root", type=Path, default=Path("Images"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet",
                        help="'none' is for offline code checks, not the pretrained baseline.")
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.threads) <= 0 or args.workers < 0:
        parser.error("epochs, batch-size, patience and threads must be positive; workers >= 0.")
    if not 0 < args.lr < float("inf"):
        parser.error("lr must be finite and positive.")
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    labels = load_labels(args.data_dir)
    train = ImageTable(args.data_dir, args.image_root, "train", labels,
                       augment=True, verify=args.verify_images)
    val = ImageTable(args.data_dir, args.image_root, "val", labels,
                     verify=args.verify_images)
    check_disjoint(train, val)
    print(f"Train={len(train)}, Validation={len(val)}, classes={len(labels)}", flush=True)
    if args.check_data_only:
        print("Data checks passed. Test CSV was not opened; no weights downloaded.")
        return
    device = choose_device(args.device)
    out = args.output_dir or Path("runs/image") / datetime.now(timezone.utc).strftime(
        "mobilenet_v2_%Y%m%dT%H%M%S_%fZ")
    # Each experiment has its own directory; never silently overwrite another run.
    out.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update({
        "architecture": "mobilenet_v2", "preprocessing": "IMAGENET1K_V2.transforms",
        "label_names": labels, "device_used": str(device),
        "selection": "highest validation macro_f1; tie -> lower validation loss",
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "torchvision": str(torchvision.__version__)},
        "data_sha256": {name: digest(args.data_dir / name) for name in [
            "train_processed.csv", "val_processed.csv", "label_mapping.json"]},
    })
    save_json(out / "config.json", config)
    try:
        model = build_model(args.weights == "imagenet", args.train_backbone).to(device)
    except Exception as exc:
        raise RuntimeError("Cannot initialize MobileNetV2/ImageNet weights. "
                           "Check network/cache. Do not silently replace pretrained weights "
                           "with random weights for a research run.") from exc
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              generator=generator, num_workers=args.workers,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    history, best_f1, best_loss, stale, best_epoch = [], -1.0, float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        train_m, *_ = run_epoch(model, train_loader, device, optimizer,
                                freeze_backbone=not args.train_backbone)
        val_m, *_ = run_epoch(model, val_loader, device)
        row = {"epoch": epoch, **{"train_" + k: v for k, v in train_m.items()},
               **{"val_" + k: v for k, v in val_m.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        improved = val_m["macro_f1"] > best_f1 or (
            val_m["macro_f1"] == best_f1 and val_m["loss"] < best_loss)
        if improved:
            best_f1, best_loss, best_epoch, stale = val_m["macro_f1"], val_m["loss"], epoch, 0
            checkpoint = {"model_state": model.state_dict(), "label_names": labels,
                          "epoch": epoch, "val_metrics": val_m,
                          "weights_initialization": args.weights,
                          "train_backbone": args.train_backbone,
                          "config": config}
            temporary = out / "best.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(out / "best.pt")
        else:
            stale += 1
        print(f"Epoch {epoch}: train_loss={train_m['loss']:.4f}, "
              f"val_loss={val_m['loss']:.4f}, val_f1={val_m['macro_f1']:.4f}", flush=True)
        if stale >= args.patience:
            break
    save_json(out / "training_summary.json", {
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "best_val_loss": best_loss, "epochs_completed": len(history),
        "test_used": False, "checkpoint": "best.pt",
    })
    print(f"Saved run: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()

