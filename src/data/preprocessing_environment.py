"""Pipeline tiền xử lý dữ liệu môi trường – độc lập với pipeline ảnh–văn bản.

Dữ liệu môi trường đến từ nguồn riêng, không suy ra từ nhãn bệnh PlantVillage,
đường dẫn ảnh, hay ngân hàng câu.

Schema đầu vào (xem environment_schema.md):
  Bắt buộc: env_id, crop, temperature (°C), humidity (%), source, split
  Tùy chọn: recorded_at, location, season

Sử dụng:
  python preprocessing_environment.py --input data/environment/environment.csv \\
                                      --output-dir data/environment
"""
import argparse
from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


REQUIRED_COLUMNS = {"env_id", "crop", "temperature", "humidity", "source"}
OPTIONAL_COLUMNS = {"recorded_at", "location", "season", "split"}
NUMERIC_COLUMNS = ["temperature", "humidity"]
SPLITS = ("train", "val", "test")


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


TEXT_REQUIRED_COLUMNS = ["env_id", "crop", "source"]


def validate_env_data(df):
    """Kiểm tra dữ liệu môi trường. Trả về danh sách lỗi (rỗng nếu  hợp lệ).

    Không tự điền số ngẫu nhiên hoặc tự gán mùa khi trường thiếu.
    """
    errors = []

    # Kiểm tra cột bắt buộc
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Thiếu cột bắt buộc: {sorted(missing)}")
        return errors  # Không kiểm tra tiếp nếu thiếu cột quan trọng

    # Kiểm tra env_id, crop, source: giá trị thiếu và chuỗi rỗng sau khi strip khoảng trắng
    for col in TEXT_REQUIRED_COLUMNS:
        if col in df.columns:
            if df[col].isna().any():
                n_missing = int(df[col].isna().sum())
                errors.append(f"Cột '{col}' có {n_missing} giá trị thiếu (NaN).")
            non_na = df[col].dropna()
            empty_mask = non_na.astype(str).str.strip() == ""
            if empty_mask.any():
                n_empty = int(empty_mask.sum())
                errors.append(f"Cột '{col}' có {n_empty} giá trị rỗng hoặc chỉ chứa khoảng trắng sau khi strip.")

    # Kiểm tra env_id trùng sau khi chuẩn hóa khoảng trắng
    if "env_id" in df.columns:
        norm_env_id = df["env_id"].dropna().astype(str).str.strip()
        norm_env_id_non_empty = norm_env_id[norm_env_id != ""]
        if norm_env_id_non_empty.duplicated().any():
            dups = norm_env_id_non_empty[norm_env_id_non_empty.duplicated()].unique().tolist()
            errors.append(f"env_id trùng lặp sau khi chuẩn hóa khoảng trắng: {dups[:5]}{'...' if len(dups) > 5 else ''}")

    # Kiểm tra ô trống trong cột số bắt buộc
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].isna().any():
            n_missing = int(df[col].isna().sum())
            errors.append(f"Cột '{col}' có {n_missing} giá trị thiếu. "
                          "Không tự điền giá trị ngẫu nhiên.")

    # Kiểm tra kiểu dữ liệu số cho temperature và humidity
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        try:
            values = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError):
            errors.append(f"Cột '{col}' chứa giá trị không phải số.")
            continue

        # Kiểm tra số hữu hạn
        if not np.isfinite(values.dropna()).all():
            n_inf = int((~np.isfinite(values.dropna())).sum())
            errors.append(f"Cột '{col}' có {n_inf} giá trị không hữu hạn (inf/nan).")

    # Kiểm tra humidity trong [0, 100]
    if "humidity" in df.columns and df["humidity"].notna().any():
        hum = pd.to_numeric(df["humidity"], errors="coerce")
        valid_hum = hum.dropna()
        if len(valid_hum) > 0:
            out_of_range = ~valid_hum.between(0, 100)
            if out_of_range.any():
                n_bad = int(out_of_range.sum())
                errors.append(f"Humidity ngoài khoảng [0, 100]: {n_bad} giá trị. "
                              "Đơn vị phải là % tương đối (0–100).")

    # Kiểm tra split nếu có
    if "split" in df.columns:
        if df["split"].isna().any():
            n_na = int(df["split"].isna().sum())
            errors.append(f"Cột 'split' có {n_na} dòng thiếu giá trị (NaN).")
        non_na_split = df["split"].dropna()
        ws_mask = non_na_split.astype(str).str.strip() == ""
        if ws_mask.any():
            n_ws = int(ws_mask.sum())
            errors.append(f"Cột 'split' có {n_ws} dòng chứa split chỉ có khoảng trắng.")
        stripped_split = non_na_split.astype(str).str.strip()
        invalid_mask = (~stripped_split.isin(SPLITS)) & (~ws_mask)
        if invalid_mask.any():
            bad_splits = sorted(set(stripped_split[invalid_mask]))
            errors.append(f"Cột 'split' chứa giá trị ngoài {SPLITS}: {bad_splits}.")

    # Kiểm tra season nếu có
    if "season" in df.columns:
        season_series = df["season"]
        is_missing_season = (
            season_series.isna()
            | (season_series.astype(str).str.strip() == "")
            | season_series.astype(str).str.strip().str.lower().isin(["nan", "none", "null"])
        )
        n_missing = int(is_missing_season.sum())
        n_present = len(df) - n_missing
        if n_missing > 0 and n_present > 0:
            errors.append(
                f"Cột 'season' có {n_missing} dòng thiếu thông tin mùa trong khi {n_present} dòng khác có. "
                "Báo lỗi để người dùng xử lý, không tự điền."
            )

    return errors


def preprocess_env(df):
    """Tiền xử lý dữ liệu môi trường: chuẩn hóa số, mã hóa season nếu có.

    Yêu cầu cột 'split' để xác định tập Train cho fitting.
    Không dùng split của pipeline ảnh–văn bản.

    Trả về dict:
      parts: {split: DataFrame đã chuẩn hóa}
      scaler: MinMaxScaler đã fit trên Train
      season_encoder: OneHotEncoder nếu có cột season, None nếu không
      report: dict thông tin xử lý
    """
    if "split" not in df.columns:
        raise ValueError(
            "Dữ liệu môi trường cần cột 'split' để xác định tập Train cho fitting. "
            "Không dùng split của pipeline ảnh–văn bản để chia bảng môi trường. "
            "Hãy xác định cách chia phù hợp cho nguồn dữ liệu môi trường trước khi fit."
        )

    # Validate trước
    errors = validate_env_data(df)
    if errors:
        raise ValueError("Dữ liệu môi trường không hợp lệ:\n" + "\n".join(f"  - {e}" for e in errors))

    stripped_splits = df["split"].dropna().astype(str).str.strip()
    valid_splits = set(stripped_splits) & set(SPLITS)
    if "train" not in valid_splits:
        raise ValueError("Cần ít nhất tập 'train' để fit scaler/encoder.")

    parts = {}
    for s in SPLITS:
        mask = df["split"].astype(str).str.strip() == s
        if mask.any():
            parts[s] = df[mask].copy().reset_index(drop=True)

    # Kiểm tra tổng số dòng đầu ra bằng đầu vào để không âm thầm bỏ dòng
    total_output_rows = sum(len(p) for p in parts.values())
    if total_output_rows != len(df):
        raise ValueError(
            f"Tổng số dòng đầu ra ({total_output_rows}) không bằng tổng số dòng đầu vào ({len(df)}). "
            "Có dòng bị bỏ do split không hợp lệ hoặc thiếu."
        )

    train = parts["train"]

    # Fit MinMaxScaler trên Train
    scaler = MinMaxScaler(feature_range=(0, 1), clip=True).fit(train[NUMERIC_COLUMNS])

    # Transform tất cả các split
    norm_cols = [f"norm_{c}" for c in NUMERIC_COLUMNS]
    for split, part in parts.items():
        part[norm_cols] = scaler.transform(part[NUMERIC_COLUMNS])

    # Mã hóa season nếu có
    season_encoder = None
    season_report = {}
    if "season" in df.columns:
        season_series = df["season"]
        is_missing_overall = (
            season_series.isna()
            | (season_series.astype(str).str.strip() == "")
            | season_series.astype(str).str.strip().str.lower().isin(["nan", "none", "null"])
        )
        if is_missing_overall.all():
            # Không có thông tin mùa ở bất kỳ dòng nào -> bỏ qua mã hóa mùa
            season_report = {
                "has_season": False,
                "season_encoding": "skipped",
                "season_note": "Không có thông tin mùa ở bất kỳ dòng nào. Bỏ qua mã hóa mùa.",
            }
        else:
            # Kiểm tra riêng tập Train trước khi fit encoder
            train_season = train["season"]
            train_missing = (
                train_season.isna()
                | (train_season.astype(str).str.strip() == "")
                | train_season.astype(str).str.strip().str.lower().isin(["nan", "none", "null"])
            )
            if train_missing.all():
                raise ValueError(
                    "Cột 'season' được cung cấp nhưng toàn bộ tập Train thiếu thông tin mùa. "
                    "Không thể fit encoder trên Train. Báo lỗi rõ ràng trước khi fit encoder."
                )
            if train_missing.any():
                raise ValueError(
                    f"Tập Train có {int(train_missing.sum())} dòng thiếu thông tin mùa. "
                    "Báo lỗi để người dùng xử lý, không tự điền."
                )

            # Chuẩn hóa khoảng trắng cho giá trị mùa trước khi fit
            train_season_clean = train[["season"]].copy()
            train_season_clean["season"] = train_season_clean["season"].astype(str).str.strip()

            season_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(train_season_clean[["season"]])
            categories = season_encoder.categories_[0].tolist()

            # Không tạo season_nan hoặc season_None
            for cat in categories:
                if str(cat).lower() in ("nan", "none", "null", ""):
                    raise ValueError(f"Category mùa không hợp lệ: '{cat}'. Không tạo season_nan hoặc season_None.")

            season_cols = [f"season_{c}" for c in categories]
            for split, part in parts.items():
                part_season_clean = part[["season"]].copy()
                part_season_clean["season"] = part_season_clean["season"].astype(str).str.strip()
                encoded = pd.DataFrame(
                    season_encoder.transform(part_season_clean[["season"]]),
                    columns=season_cols,
                    index=part.index
                )
                parts[split] = pd.concat([part, encoded], axis=1)

            season_report = {
                "has_season": True,
                "season_encoding": "one_hot",
                "season_categories": categories,
                "season_columns": season_cols,
                "season_note": f"Mã hóa {len(categories)} mùa: {categories}",
            }
    else:
        season_report = {
            "has_season": False,
            "season_encoding": "skipped",
            "season_note": "Không có cột season. Bỏ qua mã hóa mùa.",
        }

    # Report
    report = {
        "rows": len(df),
        "total_output_rows": total_output_rows,
        "splits": {s: len(p) for s, p in parts.items()},
        "numeric_columns": NUMERIC_COLUMNS,
        "scaler_fit_split": "train",
        "scaler_data_min": scaler.data_min_.tolist(),
        "scaler_data_max": scaler.data_max_.tolist(),
        "crops": sorted(df["crop"].dropna().astype(str).str.strip().unique().tolist()),
    }
    report.update(season_report)

    return {"parts": parts, "scaler": scaler, "season_encoder": season_encoder, "report": report}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV dữ liệu môi trường (xem environment_schema.md)")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Chưa có dữ liệu môi trường tại {args.input}.")
        print("Pipeline môi trường cần CSV thực với ít nhất các cột: "
              f"{sorted(REQUIRED_COLUMNS)} và cột 'split'.")
        print("Xem environment_schema.md để biết schema đầu vào.")
        return

    df = pd.read_csv(args.input, encoding="utf-8-sig")

    if df.empty:
        print(f"File {args.input} không có dòng dữ liệu.")
        print("Cần bổ sung dữ liệu môi trường từ nguồn có căn cứ trước khi xử lý.")
        return

    # Validate
    errors = validate_env_data(df)
    if errors:
        print("Dữ liệu môi trường có lỗi:")
        for e in errors:
            print(f"  - {e}")
        return

    # Preprocess
    result = preprocess_env(df)
    parts = result["parts"]
    scaler = result["scaler"]
    season_encoder = result["season_encoder"]
    report = result["report"]

    out = args.output_dir
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # Lưu artifact riêng biệt, tránh nhầm với label encoder ảnh–văn bản
    joblib.dump(scaler, artifacts / "env_scaler.pkl")
    if season_encoder is not None:
        joblib.dump(season_encoder, artifacts / "env_season_encoder.pkl")

    save_json(artifacts / "env_feature_schema.json", {
        "numeric_columns": NUMERIC_COLUMNS,
        "normalized_columns": [f"norm_{c}" for c in NUMERIC_COLUMNS],
        "season_encoded": season_encoder is not None,
        "season_columns": [f"season_{c}" for c in season_encoder.categories_[0]] if season_encoder else [],
        "scaler_fit_split": "train",
        "note": "Pipeline môi trường độc lập. Scaler/encoder fit chỉ trên tập Train của dữ liệu môi trường.",
    })

    save_json(out / "env_preprocessing_report.json", report)

    for split, part in parts.items():
        part.to_csv(out / f"{split}_env_processed.csv", index=False, encoding="utf-8-sig")
        print(f"{split}: {len(part)} bản ghi môi trường")

    print("Pipeline môi trường hoàn tất. Scaler/encoder fit chỉ trên Train.")


if __name__ == "__main__":
    main()
