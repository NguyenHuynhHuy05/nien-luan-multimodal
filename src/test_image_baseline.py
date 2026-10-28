"""Technical tests with generated images only. No NASA, PlantVillage or weights download."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from image_baseline.common import ImageTable, load_labels, build_model, check_disjoint, run_epoch


class ImageBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data, self.images = self.root / "data", self.root / "Images"
        self.data.mkdir()
        self.labels = [f"crop___class{i:02d}" for i in range(12)]
        (self.data / "label_mapping.json").write_text(json.dumps(dict(enumerate(self.labels))))
        for split in ["train", "val", "test"]:
            rows = []
            for i, label in enumerate(self.labels):
                name = f"{label}/{split}.png"
                path = self.images / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (64, 64), (i * 17, 80, 130)).save(path)
                rows.append({"image_path": name, "label": label, "label_id": i,
                             "split": split,
                             "group_id": hashlib.sha256(name.encode()).hexdigest()})
            pd.DataFrame(rows).to_csv(self.data / f"{split}_processed.csv", index=False)
        torch.set_num_threads(2)

    def tearDown(self):
        self.temp.cleanup()

    def table(self, split="val", **kwargs):
        return ImageTable(self.data, self.images, split, load_labels(self.data), **kwargs)

    def edit(self, split, column, value):
        path = self.data / f"{split}_processed.csv"
        frame = pd.read_csv(path)
        frame.loc[0, column] = value
        frame.to_csv(path, index=False)

    def test_rgb_shape_labels_and_deterministic_validation(self):
        dataset = self.table(verify=True)
        image, label = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(label, 0)
        self.assertTrue(torch.isfinite(image).all())
        self.assertTrue(torch.equal(image, dataset[0][0]))

    def test_metadata_is_not_model_input(self):
        path = self.data / "val_processed.csv"
        frame = pd.read_csv(path)
        frame["description"] = "class name that must not be used"
        frame["temperature"] = 999
        frame.to_csv(path, index=False)
        self.assertEqual(len(self.table()[0]), 2)

    def test_mapping_mismatch_rejected(self):
        self.edit("val", "label_id", 1)
        with self.assertRaisesRegex(ValueError, "mapping"):
            self.table()

    def test_missing_or_corrupt_image_not_silently_dropped(self):
        dataset = self.table()
        dataset.paths[0].write_bytes(b"invalid image")
        with self.assertRaisesRegex(ValueError, "decode"):
            self.table(verify=True)
        with self.assertRaisesRegex(RuntimeError, "load image"):
            dataset[0]
        dataset.paths[0].unlink()
        with self.assertRaises(FileNotFoundError):
            self.table()

    def test_path_traversal_rejected(self):
        self.edit("val", "image_path", "../outside.png")
        with self.assertRaises(ValueError):
            self.table()

    def test_group_overlap_rejected(self):
        train = self.table("train")
        self.edit("val", "group_id", train.frame.iloc[0]["group_id"])
        with self.assertRaisesRegex(ValueError, "group_id"):
            check_disjoint(train, self.table())

    def test_head_updates_but_frozen_backbone_and_batchnorm_do_not(self):
        torch.manual_seed(42)
        model = build_model(pretrained=False)
        feature = next(model.features.parameters()).detach().clone()
        bn = next(module for module in model.features.modules()
                  if isinstance(module, torch.nn.BatchNorm2d))
        running_mean = bn.running_mean.clone()
        head = model.classifier[1].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.01)
        run_epoch(model, DataLoader(self.table("train"), batch_size=12),
                  torch.device("cpu"), optimizer)
        self.assertTrue(torch.equal(feature, next(model.features.parameters())))
        self.assertTrue(torch.equal(running_mean, bn.running_mean))
        self.assertFalse(torch.equal(head, model.classifier[1].weight))

    def test_train_save_load_and_evaluate_without_reading_test_during_training(self):
        source = Path(__file__).resolve().parent
        test_csv = self.data / "test_processed.csv"
        backup = test_csv.read_bytes()
        test_csv.unlink()
        run = self.root / "run"
        args = ["--data-dir", str(self.data), "--image-root", str(self.images),
                "--batch-size", "12", "--device", "cpu", "--threads", "2"]
        env = os.environ.copy()
        def cli(script, more):
            result = subprocess.run([sys.executable, str(source / script)] + args + more,
                                    capture_output=True, text=True, timeout=90, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        cli("train_image.py", ["--epochs", "1", "--weights", "none", "--output-dir", str(run)])
        checkpoint = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
        self.assertEqual(checkpoint["label_names"], self.labels)
        summary = json.loads((run / "training_summary.json").read_text())
        self.assertFalse(summary["test_used"])
        self.assertEqual(len(pd.read_csv(run / "history.csv")), 1)
        cli("evaluate_image.py", ["--checkpoint", str(run / "best.pt"), "--split", "val"])
        test_csv.write_bytes(backup)
        cli("evaluate_image.py", ["--checkpoint", str(run / "best.pt"), "--split", "test"])
        metrics = json.loads((run / "evaluation_test/metrics.json").read_text())
        self.assertEqual(metrics["rows"], 12)
        self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 12)
        preds = pd.read_csv(run / "evaluation_test/predictions.csv")
        self.assertEqual(preds["label_id"].tolist(), list(range(12)))
        self.assertEqual(len(preds), 12)


if __name__ == "__main__":
    unittest.main()

