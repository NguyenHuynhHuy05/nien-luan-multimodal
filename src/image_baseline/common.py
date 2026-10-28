"""Shared data/model/metrics code for the image-only baseline."""
import hashlib
import json
import random
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

WEIGHTS = MobileNet_V2_Weights.IMAGENET1K_V2


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Use --device cpu or install the matching CUDA build.")
    return torch.device(name)


def load_labels(data_dir):
    raw = json.loads((Path(data_dir) / "label_mapping.json").read_text(encoding="utf-8-sig"))
    expected = {str(i) for i in range(12)}
    if set(raw) != expected or len(set(raw.values())) != 12:
        raise ValueError("label_mapping.json must map IDs 0..11 to 12 unique labels.")
    return [raw[str(i)] for i in range(12)]


def resolve_image(root, name):
    name = str(name).replace("\\", "/")
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or ":" in name or len(pure.parts) < 2:
        raise ValueError(f"Unsafe relative image_path: {name}")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"image_path escapes image root: {name}")
    if not path.is_file():
        raise FileNotFoundError(f"Image missing: {path}")
    return path


class ImageTable(Dataset):
    """CSV split is authoritative; pixels alone are passed to the classifier."""
    def __init__(self, data_dir, image_root, split, labels, augment=False, verify=False):
        self.root = Path(image_root).resolve()
        self.frame = pd.read_csv(Path(data_dir) / f"{split}_processed.csv", encoding="utf-8-sig")
        required = ["image_path", "label", "label_id", "split", "group_id"]
        if not set(required).issubset(self.frame):
            raise ValueError(f"CSV requires columns: {required}")
        if self.frame.empty or self.frame[required].isna().any().any():
            raise ValueError(f"{split}: empty table or missing required values.")
        if not self.frame["split"].eq(split).all():
            raise ValueError(f"{split}: rows have a different split.")
        ids = pd.to_numeric(self.frame["label_id"], errors="raise")
        if not np.isfinite(ids).all() or not ids.eq(ids.round()).all() or not ids.between(0, 11).all():
            raise ValueError(f"{split}: label_id must be integer 0..11.")
        self.frame["label_id"] = ids.astype(int)
        expected = self.frame["label_id"].map(dict(enumerate(labels)))
        if not expected.eq(self.frame["label"]).all():
            raise ValueError(f"{split}: label_id does not agree with label_mapping.json.")
        if set(self.frame["label"]) != set(labels):
            raise ValueError(f"{split}: one or more of the 12 classes is missing.")
        self.path_keys = self.frame["image_path"].str.replace("\\", "/", regex=False).str.casefold()
        if self.path_keys.duplicated().any():
            raise ValueError(f"{split}: duplicate image_path.")
        if self.frame.groupby("group_id")["label"].nunique().max() > 1:
            raise ValueError(f"{split}: a group has multiple labels.")
        self.paths = []
        for row in self.frame.itertuples():
            if PurePosixPath(str(row.image_path).replace("\\", "/")).parts[0] != row.label:
                raise ValueError(f"Folder/label mismatch: {row.image_path}")
            path = resolve_image(self.root, row.image_path)
            if verify:
                try:
                    with Image.open(path) as im:
                        im.verify()
                except Exception as exc:
                    raise ValueError(f"Cannot decode image: {path}") from exc
            self.paths.append(path)
        # Val/Test use exactly the preprocessing specified by these weights.
        self.transform = (transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            WEIGHTS.transforms(),
        ]) if augment else WEIGHTS.transforms())

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Cannot load image: {path}") from exc
        return tensor, int(self.frame.iloc[index]["label_id"])


def check_disjoint(a, b):
    if set(a.path_keys) & set(b.path_keys):
        raise ValueError("image_path overlap between splits.")
    if set(a.frame["group_id"]) & set(b.frame["group_id"]):
        raise ValueError("group_id overlap between splits.")


def build_model(pretrained=True, train_backbone=False):
    model = mobilenet_v2(weights=WEIGHTS if pretrained else None)
    for parameter in model.features.parameters():
        parameter.requires_grad = train_backbone
    model.classifier[1] = nn.Linear(model.last_channel, 12)
    return model


def run_epoch(model, loader, device, optimizer=None, freeze_backbone=True):
    training = optimizer is not None
    model.train(training)
    # Frozen BatchNorm statistics must not change during head-only training.
    if training and freeze_backbone:
        model.features.eval()
    total, loss_sum, truth, predicted, scores = 0, 0.0, [], [], []
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; training stopped.")
            if training:
                loss.backward()
                optimizer.step()
            prob = logits.detach().softmax(dim=1)
            confidence, classes = prob.max(dim=1)
            total += len(targets)
            loss_sum += float(loss.detach()) * len(targets)
            truth.extend(targets.cpu().tolist())
            predicted.extend(classes.cpu().tolist())
            scores.extend(confidence.cpu().tolist())
    if total == 0:
        raise ValueError("Empty loader.")
    metrics = {
        "rows": total,
        "loss": loss_sum / total,
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=list(range(12)),
                                   average="macro", zero_division=0)),
    }
    return metrics, truth, predicted, scores


def detailed_metrics(truth, predicted, labels):
    ids = list(range(len(labels)))
    return {
        "class_order": labels,
        "classification_report": classification_report(
            truth, predicted, labels=ids, target_names=labels,
            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=ids).tolist(),
        "confusion_matrix_axes": "rows=true, columns=predicted",
    }

