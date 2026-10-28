"""Audit exact image bytes across existing splits without changing research data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import pandas as pd
from PIL import Image

SPLITS = ("train", "val", "test")


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def collect_manifest(data_dir, image_root):
    root = Path(image_root).resolve()
    records = []
    for split in SPLITS:
        table = Path(data_dir) / f"{split}_processed.csv"
        frame = pd.read_csv(table, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if frame.empty or not {"image_path", "split"}.issubset(frame.columns):
            raise ValueError(f"{table}: expected nonempty image_path and split columns")
        for row in frame.to_dict("records"):
            name = row["image_path"].replace("\\", "/")
            pure = PurePosixPath(name)
            if row["split"] != split:
                raise ValueError(f"{table}: incorrect split for {name}")
            if not name or pure.is_absolute() or ".." in pure.parts or ":" in name:
                raise ValueError(f"Unsafe image_path: {name!r}")
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Image escapes root: {name}")
            if not path.is_file():
                raise FileNotFoundError(f"Image missing: {path}")
            try:
                digest = sha256_file(path)
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
            except (OSError, ValueError, SyntaxError) as exc:
                raise ValueError(f"Cannot read/decode image [{split}] {path}: {exc}") from exc
            records.append({"split": split, "image_path": name, "sha256": digest})
    keys = [(row["split"], row["image_path"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate image_path rows within a split")
    return records


def cross_split_duplicates(records):
    groups = {}
    for row in records:
        groups.setdefault(row["sha256"], []).append(row)
    return {digest: rows for digest, rows in sorted(groups.items())
            if len({row["split"] for row in rows}) > 1}


def compare_manifest(records, previous):
    def index(rows):
        result = {}
        for row in rows:
            key = (row["split"], row["image_path"])
            digest = row["sha256"]
            if row["split"] not in SPLITS or not isinstance(digest, str) or len(digest) != 64 or any(
                    char not in "0123456789abcdef" for char in digest):
                raise ValueError("Invalid manifest record")
            if key in result:
                raise ValueError(f"Duplicate manifest key: {key}")
            result[key] = digest
        return result
    current, old = index(records), index(previous)
    return {
        "added": sorted(set(current) - set(old)),
        "removed": sorted(set(old) - set(current)),
        "changed": sorted(key for key in set(current) & set(old) if current[key] != old[key]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/image_text"))
    parser.add_argument("--image-root", type=Path, default=Path("Images"))
    parser.add_argument("--output", type=Path, required=True, help="New manifest CSV; never overwritten")
    parser.add_argument("--compare", type=Path, help="Previously saved manifest CSV")
    args = parser.parse_args()
    try:
        # Refuse to overwrite a baseline, source CSV, or image.
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}; choose a new path")
        records = collect_manifest(args.data_dir, args.image_root)
        changes = None
        if args.compare:
            with args.compare.open(encoding="utf-8-sig", newline="") as stream:
                changes = compare_manifest(records, list(csv.DictReader(stream)))
        duplicates = cross_split_duplicates(records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["split", "image_path", "sha256"])
            writer.writeheader()
            writer.writerows(records)
        print(f"Hashed {len(records)} CSV-referenced images. Manifest: {args.output.resolve()}")
        if duplicates:
            print("STOP: identical image bytes across splits:", file=sys.stderr)
            for digest, rows in duplicates.items():
                print(f"SHA256 {digest}", file=sys.stderr)
                for row in rows:
                    print(f"  [{row['split']}] {row['image_path']}", file=sys.stderr)
        if changes is not None:
            print("Manifest comparison: " + json.dumps(changes, ensure_ascii=False))
        if duplicates or (changes is not None and any(changes.values())):
            return 1
        print("No cross-split byte duplicates." if changes is None else
              "No cross-split byte duplicates; manifest matches baseline.")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
