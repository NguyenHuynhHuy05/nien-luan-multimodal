"""Tiền xử lý pipeline ảnh–văn bản: giữ nguyên split, chỉ fit trên Train.

Không xử lý dữ liệu môi trường; xem preprocessing_environment.py cho pipeline riêng.
"""
import argparse
from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pipeline_utils import SPLITS, clean_vietnamese_text, save_json, validate_dataset


def preprocess(df):
    """Tiền xử lý bảng ảnh–văn bản: làm sạch text, mã hóa nhãn.

    Trả về (parts, label_encoder, report).
    - parts: dict {split: DataFrame} đã có clean_description và label_id.
    - label_encoder: LabelEncoder đã fit trên Train.
    - report: dict kết quả kiểm tra.
    """
    # Loại các cột môi trường cũ nếu đầu vào còn mang theo
    env_cols_to_drop = [c for c in df.columns
                        if c in ("temperature", "humidity", "season", "environment_source",
                                 "norm_temperature", "norm_humidity")
                        or c.startswith("season_")]
    if env_cols_to_drop:
        df = df.drop(columns=env_cols_to_drop)

    report = validate_dataset(df)
    parts = {s: df[df["split"] == s].copy().reset_index(drop=True) for s in SPLITS}
    train = parts["train"]
    labels = LabelEncoder().fit(train["label"])
    for split, part in parts.items():
        part["clean_description"] = part["description"].map(clean_vietnamese_text)
        part["label_id"] = labels.transform(part["label"])
    return parts, labels, report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    parts, labels, report = preprocess(df)
    out = args.output_dir
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    save_json(out / "label_mapping.json", {int(i): label for i, label in enumerate(labels.classes_)})
    joblib.dump(labels, artifacts / "label_encoder.pkl")
    save_json(artifacts / "feature_schema.json", {
        "image_root": "Images", "image_path_column": "image_path",
        "text_column": "clean_description", "label_column": "label_id",
        "input_features": ["image_path", "clean_description"],
        "target": "label_id",
        "exclude_from_model": ["label", "split", "group_id", "text_level",
                               "text_family_id", "text_source", "description"],
        "label_encoder_fit_split": "train",
        "note": "Pipeline ảnh–văn bản. Dữ liệu môi trường được xử lý riêng trong pipeline_environment.",
    })
    save_json(out / "preprocessing_report.json", report)
    for split, part in parts.items():
        part.to_csv(out / f"{split}_processed.csv", index=False, encoding="utf-8-sig")
        print(f"{split}: {len(part)} mẫu")
    print("Giữ nguyên split; label encoder fit chỉ trên Train.")


if __name__ == "__main__":
    main()
