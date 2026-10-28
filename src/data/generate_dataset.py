"""Sinh bảng ảnh–văn bản từ ảnh PlantVillage; không sinh dữ liệu môi trường.

python generate_dataset.py --data-dir Images
python generate_dataset.py --manifest backup_v1/dataset.csv
Chế độ manifest chỉ tái sử dụng image_path + label, không dùng text/env V1.
"""
import argparse
from pathlib import Path, PurePosixPath
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline_utils import SPLITS, path_key, sha256_file, save_json, validate_dataset
from text_bank import PROFILES, make_text_bank


def load_inventory(data_dir=None, manifest=None):
    if manifest is not None:
        df = pd.read_csv(manifest, encoding="utf-8-sig", dtype=str)
        if not {"image_path", "label"}.issubset(df.columns):
            raise ValueError("Manifest phải có image_path và label.")
        # Bỏ hoàn toàn description, environment, split và metadata cũ.
        df = df[["image_path", "label"]].copy()
        mode = "manifest_paths_only"
    else:
        root = Path(data_dir)
        if not root.is_dir():
            raise ValueError(f"Không thấy thư mục {root}. Dùng --data-dir hoặc --manifest.")
        records = []
        for label in sorted(PROFILES):
            folder = root / label
            if not folder.is_dir():
                raise ValueError(f"Thiếu thư mục lớp: {folder}")
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    records.append({"image_path": path.relative_to(root).as_posix(),
                                    "label": label, "group_id": sha256_file(path)})
        df = pd.DataFrame(records)
        mode = "sha256_image_bytes"
    if df.empty or df[["image_path", "label"]].isna().any().any():
        raise ValueError("Danh sách ảnh rỗng hoặc có ô thiếu.")
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    for path in df["image_path"]:
        p = PurePosixPath(path)
        if p.is_absolute() or ".." in p.parts or ":" in path or len(p.parts) < 2:
            raise ValueError(f"Đường dẫn phải tương đối dưới Images: {path}")
    if not all(PurePosixPath(p).parts[0] == label for p, label in zip(df["image_path"], df["label"])):
        raise ValueError("Thư mục đầu tiên trong image_path không khớp label.")
    if set(df["label"]) != set(PROFILES):
        raise ValueError(f"Phải có đúng 12 lớp đã chọn. Thiếu: {set(PROFILES) - set(df['label'])}; lạ: {set(df['label']) - set(PROFILES)}")
    if df["image_path"].map(path_key).duplicated().any():
        raise ValueError("Manifest có đường dẫn trùng.")
    if "group_id" not in df:
        df["group_id"] = df["image_path"].map(path_key)
    if df.groupby("group_id")["label"].nunique().max() > 1:
        raise ValueError("Ảnh trùng byte nhưng mang nhiều nhãn. Cần rà soát trước khi split.")
    return df.sort_values("image_path").reset_index(drop=True), mode


def split_inventory(df, seed):
    # Khi đọc ảnh thật, các ảnh trùng byte luôn đi cùng một split.
    groups = df[["group_id", "label"]].drop_duplicates().reset_index(drop=True)
    if groups["label"].value_counts().min() < 7:
        raise ValueError("Mỗi lớp cần ít nhất 7 nhóm ảnh để chia stratified 70/15/15.")
    train, temp = train_test_split(groups, test_size=0.3, stratify=groups["label"], random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=seed)
    assigned = pd.concat([part.assign(split=split) for part, split in zip((train, val, test), SPLITS)])
    return df.merge(assigned[["group_id", "split"]], on="group_id", validate="many_to_one")


IMAGE_TEXT_COLUMNS = ["image_path", "label", "group_id", "split",
                      "description", "text_level", "text_family_id", "text_source"]


def synthesize(df, seed):
    """Sinh mô tả văn bản cho từng ảnh; không sinh dữ liệu môi trường.

    Đầu ra luôn chỉ có đúng 8 cột IMAGE_TEXT_COLUMNS, bất kể đầu vào có thêm cột nào.
    """
    base_cols = [c for c in ["image_path", "label", "group_id", "split"] if c in df.columns]
    df = df[base_cols].copy()
    bank = make_text_bank()
    text_rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    records = []
    for split in SPLITS:
        for label in sorted(PROFILES):
            part = df[(df["split"] == split) & (df["label"] == label)].sort_values("image_path")
            n = len(part)
            # Tỷ lệ gần 40/40/20 trong từng lớp và split, không gắn độ đầy đủ với nhãn.
            detailed_n, partial_n = round(n * 0.4), round(n * 0.4)
            levels = np.array(["detailed"] * detailed_n + ["partial"] * partial_n + ["generic"] * (n - detailed_n - partial_n))
            text_rng.shuffle(levels)
            for row, level in zip(part.to_dict("records"), levels):
                entries = bank[split][label][level]
                entry = entries[int(text_rng.integers(len(entries)))]
                row.update(description=entry["description"], text_level=str(level),
                           text_family_id=entry["family_id"], text_source="synthetic_label_conditioned")
                records.append(row)
    result = pd.DataFrame(records)
    return result[IMAGE_TEXT_COLUMNS], bank


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-dir", type=Path, help="Thư mục Images chứa 12 lớp")
    source.add_argument("--manifest", type=Path, help="CSV cũ; chỉ lấy image_path + label")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    inventory, mode = load_inventory(args.data_dir or Path("Images"), args.manifest)
    df, bank = synthesize(split_inventory(inventory, args.seed), args.seed)
    report = validate_dataset(df)
    report.update(seed=args.seed, image_check_mode=mode,
                  exact_duplicate_image_rows=int(len(inventory) - inventory["group_id"].nunique()),
                  unique_image_groups=int(inventory["group_id"].nunique()),
                  target_split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
                  limitation="Text sinh từ nhãn, chưa được đối chiếu từng ảnh. Chưa kiểm tra ảnh gần trùng hoặc cùng lá/cây.")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "dataset.csv", index=False, encoding="utf-8-sig")
    save_json(out / "generation_report.json", report)
    save_json(out / "text_bank.json", bank)
    print(f"Đã sinh bảng ảnh–văn bản: {len(df)} mẫu; split: {report['split_counts']}")
    print(f"Kiểm tra ảnh: {mode}; trùng câu sau làm sạch: 0 giữa mọi cặp split.")
    if mode == "manifest_paths_only":
        print("Chưa đọc byte ảnh; chạy với --data-dir Images để kiểm tra trùng ảnh SHA-256.")


if __name__ == "__main__":
    main()
