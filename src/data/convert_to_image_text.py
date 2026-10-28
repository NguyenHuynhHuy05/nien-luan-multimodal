"""Chuyển đổi dataset.csv hiện có sang bảng ảnh–văn bản (không có cột môi trường).

Giữ nguyên image_path, label, group_id, split và toàn bộ thông tin văn bản đã có.
Chỉ loại các cột môi trường. Không chia lại ảnh hoặc sinh lại mô tả.
Sau đó chạy preprocessing ảnh–văn bản và xuất vào data/image_text/.

Sử dụng:
  cd src/data
  python convert_to_image_text.py
"""
import sys
from pathlib import Path
import pandas as pd

# Thêm src/data vào path để import module cùng thư mục
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_utils import validate_dataset, save_json
from preprocessing import preprocess


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    project_root = Path(__file__).resolve().parent.parent.parent
    source_path = project_root / "data" / "dataset.csv"
    output_dir = project_root / "data" / "image_text"

    print(f"Đọc dữ liệu nguồn: {source_path}")
    df = pd.read_csv(source_path, encoding="utf-8-sig")

    # Xác minh số liệu
    print(f"Tổng: {len(df)} ảnh")
    split_counts = df["split"].value_counts().sort_index()
    for s, c in split_counts.items():
        print(f"  {s}: {c}")

    expected = {"train": 9557, "val": 2046, "test": 2047}
    for s, e in expected.items():
        actual = split_counts.get(s, 0)
        if actual != e:
            print(f"CẢNH BÁO: Dự kiến {s}={e}, thực tế={actual}")

    # Giữ nguyên các cột ảnh–văn bản, loại cột môi trường
    image_text_cols = ["image_path", "label", "group_id", "split",
                       "description", "text_level", "text_family_id", "text_source"]

    missing_cols = set(image_text_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dữ liệu nguồn thiếu cột: {sorted(missing_cols)}")

    df_it = df[image_text_cols].copy()
    env_dropped = [c for c in df.columns if c not in image_text_cols]
    print(f"Đã loại các cột: {env_dropped}")

    # Validate bảng ảnh–văn bản
    report = validate_dataset(df_it)
    print(f"Validation ảnh–văn bản OK: {report['rows']} mẫu, {report['classes']} lớp")

    # Lưu dataset ảnh–văn bản
    output_dir.mkdir(parents=True, exist_ok=True)
    df_it.to_csv(output_dir / "dataset.csv", index=False, encoding="utf-8-sig")
    save_json(output_dir / "generation_report.json", report)
    print(f"Đã lưu bảng ảnh–văn bản: {output_dir / 'dataset.csv'}")

    # Chạy preprocessing
    parts, labels, prep_report = preprocess(df_it)

    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    import joblib
    save_json(output_dir / "label_mapping.json",
              {int(i): label for i, label in enumerate(labels.classes_)})
    joblib.dump(labels, artifacts / "label_encoder.pkl")
    save_json(artifacts / "feature_schema.json", {
        "image_root": "Images", "image_path_column": "image_path",
        "text_column": "clean_description", "label_column": "label_id",
        "input_features": ["image_path", "clean_description"],
        "target": "label_id",
        "exclude_from_model": ["label", "split", "group_id", "text_level",
                               "text_family_id", "text_source", "description"],
        "label_encoder_fit_split": "train",
        "note": "Pipeline ảnh–văn bản. Dữ liệu môi trường được xử lý riêng.",
    })
    save_json(output_dir / "preprocessing_report.json", prep_report)

    for split, part in parts.items():
        part.to_csv(output_dir / f"{split}_processed.csv", index=False, encoding="utf-8-sig")
        print(f"  {split}: {len(part)} mẫu")

    # Kiểm tra bảo toàn split và label mapping
    combined = pd.concat(parts.values(), ignore_index=True)
    source_split_map = dict(zip(df["image_path"], df["split"]))
    result_split_map = dict(zip(combined["image_path"], combined["split"]))
    assert source_split_map == result_split_map, "Split đã bị thay đổi!"

    source_label_map = dict(zip(df["image_path"], df["label"]))
    result_label_map = dict(zip(combined["image_path"], combined["label"]))
    assert source_label_map == result_label_map, "Label đã bị thay đổi!"

    print("\nChuyển đổi hoàn tất. Split, group_id, mô tả và label mapping được giữ nguyên.")
    print(f"Thư mục đầu ra: {output_dir}")


if __name__ == "__main__":
    main()
