"""Audit bảng ảnh–văn bản; không phụ thuộc cột hoặc artifact môi trường.

Baseline majority và TF-IDF được cung cấp dưới dạng tùy chọn (--baselines).
Audit cấu trúc dữ liệu luôn chạy.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import make_pipeline
from pipeline_utils import SPLITS, clean_vietnamese_text, save_json, validate_dataset


def baseline_scores(train, test):
    """Chạy baseline trên dữ liệu ảnh–văn bản; không dùng cột môi trường."""
    y_train, y_test = train["label_id"], test["label_id"]
    # Majority baseline chỉ cần nhãn, dùng cột clean_description làm X giả.
    models = {
        "majority": (DummyClassifier(strategy="most_frequent"),
                     train[["clean_description"]], test[["clean_description"]]),
        "text_tfidf_logistic": (make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                                              LogisticRegression(max_iter=1500, random_state=42)),
                                train["clean_description"], test["clean_description"]),
    }
    scores = {}
    for name, (model, x_train, x_test) in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        scores[name] = {"accuracy": float(accuracy_score(y_test, pred)),
                        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                        "per_class": classification_report(y_test, pred, output_dict=True, zero_division=0)}
        if name == "text_tfidf_logistic" and "text_level" in test:
            scores[name]["by_text_level"] = {}
            for level in ("detailed", "partial", "generic"):
                mask = test["text_level"].eq(level).to_numpy()
                if mask.sum() == 0:
                    continue
                scores[name]["by_text_level"][level] = {
                    "rows": int(mask.sum()),
                    "accuracy": float(accuracy_score(y_test[mask], pred[mask])),
                    "macro_f1": float(f1_score(y_test[mask], pred[mask], average="macro", zero_division=0)),
                }
    return scores


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--baselines", action="store_true",
                        help="Chạy baseline majority và TF-IDF (tùy chọn)")
    args = parser.parse_args()

    parts = {s: pd.read_csv(args.data_dir / f"{s}_processed.csv", encoding="utf-8-sig") for s in SPLITS}
    for split, part in parts.items():
        if not part["split"].eq(split).all():
            raise ValueError(f"File {split} có hàng thuộc split khác.")

    combined = pd.concat(parts.values(), ignore_index=True)
    report = validate_dataset(combined)

    # So sánh với bảng nguồn dataset.csv nếu có
    source_path = args.data_dir / "dataset.csv"
    if source_path.exists():
        source = pd.read_csv(source_path, encoding="utf-8-sig")
        # Loại các cột môi trường cũ nếu source còn giữ
        env_cols_to_ignore = [c for c in source.columns
                              if c in ("temperature", "humidity", "season", "environment_source",
                                       "norm_temperature", "norm_humidity")
                              or c.startswith("season_")]
        if env_cols_to_ignore:
            source = source.drop(columns=env_cols_to_ignore)
        if set(source["image_path"]) != set(combined["image_path"]):
            raise ValueError("Processed không chứa đầy đủ ảnh dataset.csv.")
        if dict(zip(source["image_path"], source["split"])) != dict(zip(combined["image_path"], combined["split"])):
            raise ValueError("Split đã bị thay đổi trong preprocessing.")
        raw = source.set_index("image_path").sort_index()
        processed = combined.set_index("image_path").sort_index()
        common_cols = [c for c in raw.columns if c in processed.columns]
        pd.testing.assert_frame_equal(raw[common_cols], processed[common_cols], check_dtype=False)

    # Kiểm tra label_encoder
    artifacts = args.data_dir / "artifacts"
    labels = joblib.load(artifacts / "label_encoder.pkl")
    for split, part in parts.items():
        np.testing.assert_array_equal(part["label_id"], labels.transform(part["label"]))
        if not part["clean_description"].equals(part["description"].map(clean_vietnamese_text)):
            raise ValueError("clean_description không khớp hàm làm sạch.")

    # Baseline tùy chọn
    if args.baselines:
        report["baselines"] = baseline_scores(parts["train"], parts["test"])

    report["interpretation"] = ("Không đặt ngưỡng Accuracy để tuyên bố hết leakage. "
                                "Text vẫn sinh từ nhãn; chưa kiểm tra trên mô tả quan sát thực tế.")
    save_json(args.data_dir / "audit_report.json", report)
    counts = combined.groupby(["label", "split"]).size().unstack(fill_value=0)[list(SPLITS)]
    counts["total"] = counts.sum(axis=1)
    counts.to_csv(args.data_dir / "class_counts.csv", encoding="utf-8-sig")
    print("Audit ảnh–văn bản hoàn tất. Trùng đường dẫn/nhóm ảnh/câu/họ câu giữa các split: 0.")
    if args.baselines and "baselines" in report:
        for name, score in report["baselines"].items():
            print(f"{name}: Accuracy={score['accuracy']:.4f}, Macro F1={score['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
