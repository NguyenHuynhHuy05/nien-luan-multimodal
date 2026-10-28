"""Evaluate a saved image model. Default is Validation; Test must be requested explicitly."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from image_baseline.common import (
    ImageTable, load_labels, check_disjoint, build_model, run_epoch,
    detailed_metrics, choose_device, save_json, digest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/image_text"))
    parser.add_argument("--image-root", type=Path, default=Path("Images"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if min(args.batch_size, args.threads) <= 0 or args.workers < 0:
        parser.error("batch-size and threads must be positive; workers >= 0.")
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    labels = load_labels(args.data_dir)
    if labels != checkpoint["label_names"]:
        raise ValueError("Label mapping differs from the saved checkpoint.")
    for name, expected in checkpoint["config"]["data_sha256"].items():
        if digest(args.data_dir / name) != expected:
            raise ValueError(f"Data changed since training: {name}")
    train = ImageTable(args.data_dir, args.image_root, "train", labels)
    selected = ImageTable(args.data_dir, args.image_root, args.split, labels)
    check_disjoint(train, selected)
    if args.split == "test":
        val = ImageTable(args.data_dir, args.image_root, "val", labels)
        check_disjoint(val, selected)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    loader = DataLoader(selected, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    metrics, truth, predicted, scores = run_epoch(model, loader, device)
    metrics.update(detailed_metrics(truth, predicted, labels))
    metrics.update({"split": args.split, "checkpoint": str(args.checkpoint.resolve()),
                    "epoch": checkpoint["epoch"],
                    "evaluated_csv_sha256": digest(args.data_dir / f"{args.split}_processed.csv")})
    out = args.output_dir or args.checkpoint.parent / f"evaluation_{args.split}"
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "metrics.json", metrics)
    frame = selected.frame[["image_path", "label", "label_id"]].copy()
    frame["predicted_id"] = predicted
    frame["predicted_label"] = [labels[i] for i in predicted]
    frame["max_softmax_score"] = scores  # Not a calibrated disease probability.
    frame.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    import pandas as pd
    pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels).to_csv(
        out / "confusion_matrix.csv", encoding="utf-8-sig")
    print(f"{args.split}: Accuracy={metrics['accuracy']:.4f}, "
          f"Macro F1={metrics['macro_f1']:.4f}; saved {out.resolve()}")


if __name__ == "__main__":
    main()

