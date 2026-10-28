"""Các kiểm tra dùng chung cho pipeline ảnh–văn bản; không đưa metadata vào mô hình."""
import hashlib
import json
import re
import unicodedata
from itertools import combinations
from pathlib import Path

SPLITS = ("train", "val", "test")


def clean_vietnamese_text(text):
    text = unicodedata.normalize("NFC", str(text).lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def path_key(path):
    return unicodedata.normalize("NFC", str(path).replace("\\", "/")).casefold()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_dataset(df):
    """Kiểm tra bảng ảnh–văn bản; không yêu cầu cột môi trường."""
    required = {"image_path", "label", "split", "description", "group_id"}
    if required - set(df.columns):
        raise ValueError(f"Thiếu cột: {sorted(required - set(df.columns))}")
    if df[list(required)].isna().any().any():
        raise ValueError("Dữ liệu có ô thiếu.")
    if set(df["split"]) != set(SPLITS):
        raise ValueError("Phải có đúng ba split: train, val, test.")
    if df["image_path"].map(path_key).duplicated().any():
        raise ValueError("Trùng đường dẫn ảnh.")
    all_labels = set(df["label"])
    clean = df["description"].map(clean_vietnamese_text)
    if clean.eq("").any():
        raise ValueError("Mô tả rỗng sau khi làm sạch.")
    if df.groupby("group_id")["label"].nunique().max() > 1:
        raise ValueError("Một nhóm ảnh có nhiều nhãn.")
    report = {"rows": len(df), "classes": len(all_labels), "split_counts": {},
              "clean_text_unique": {}, "overlap": {}}
    for split in SPLITS:
        mask = df["split"].eq(split)
        if set(df.loc[mask, "label"]) != all_labels:
            raise ValueError(f"{split} thiếu lớp.")
        report["split_counts"][split] = int(mask.sum())
        report["clean_text_unique"][split] = int(clean[mask].nunique())
    for a, b in combinations(SPLITS, 2):
        ma, mb = df["split"].eq(a), df["split"].eq(b)
        item = {
            "image_paths": len(set(df.loc[ma, "image_path"].map(path_key)) & set(df.loc[mb, "image_path"].map(path_key))),
            "image_groups": len(set(df.loc[ma, "group_id"]) & set(df.loc[mb, "group_id"])),
            "clean_descriptions": len(set(clean[ma]) & set(clean[mb])),
        }
        if "text_family_id" in df:
            item["text_families"] = len(set(df.loc[ma, "text_family_id"]) & set(df.loc[mb, "text_family_id"]))
        report["overlap"][f"{a}/{b}"] = item
        if any(item.values()):
            raise ValueError(f"Có trùng giữa {a}/{b}: {item}")
    return report
