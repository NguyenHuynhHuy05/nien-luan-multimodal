"""Dịch vụ lấy dữ liệu môi trường từ NASA POWER API (temporal/daily/point).

Chỉ lấy T2M (nhiệt độ 2m, °C) và RH2M (độ ẩm tương đối 2m, %) theo ngày.
Không suy ra crop, season hay nhãn bệnh. Không chạy scaler hay encoder.
Độc lập hoàn toàn với pipeline ảnh–văn bản.

Cấu trúc phản hồi API (xác minh từ phản hồi thực tế 2024-01-01 → 2024-01-03):
  {
    "type": "Feature",
    "geometry": {...},
    "properties": {
      "parameter": {
        "T2M":  {"YYYYMMDD": value, ...},
        "RH2M": {"YYYYMMDD": value, ...}
      }
    },
    "header": {
      "fill_value": -999.0,
      "time_standard": "LST",
      "start": "YYYYMMDD",
      "end": "YYYYMMDD"
    },
    "parameters": {
      "T2M":  {"units": "C",  "longname": "Temperature at 2 Meters"},
      "RH2M": {"units": "%",  "longname": "Relative Humidity at 2 Meters"}
    },
    "messages": []
  }

Mã fill_value = -999.0 nghĩa là dữ liệu bị thiếu tại ngày đó.
Bản ghi hợp lệ: T2M != fill_value, RH2M in [0, 100], cả hai là số hữu hạn.

Sử dụng CLI:
  python nasa_power_service.py \\
      --latitude 21.028 --longitude 105.85 \\
      --start 2024-01-01 --end 2024-01-03 \\
      --output-dir ../../data/environment/nasa_power
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_API_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
_PARAMETERS   = "T2M,RH2M"
_COMMUNITY    = "AG"
_FORMAT       = "JSON"
_TIME_STD     = "LST"
_SOURCE_NAME  = "NASA_POWER"
_TIMEOUT_SEC  = 60
_FILL_VALUE   = -999.0      # gia tri "thieu" theo tai lieu va phan hoi thuc te
_FILL_TOL     = 0.5         # nguong so sanh voi fill_value

# Don vi xac minh tu phan hoi thuc te
_EXPECTED_UNITS = {"T2M": "C", "RH2M": "%"}

# Cot dau ra
OUTPUT_COLUMNS = [
    "env_id", "latitude", "longitude", "date",
    "temperature", "humidity",
    "source", "time_standard", "data_status",
]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> date:
    """Chuyen chuoi YYYY-MM-DD thanh date. Nem ValueError neu sai dinh dang."""
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(
            f"Ngay '{date_str}' khong dung dinh dang YYYY-MM-DD. "
            "Vi du: 2024-01-15"
        )


def _validate_inputs(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
) -> None:
    """Kiem tra toa do va khoang ngay. Nem ValueError neu khong hop le.

    Luu y: bool la subclass cua int nen phai kiem tra truoc isinstance(x, (int, float)).
    """
    # --- Kiem tra Boolean truoc khi kiem tra so ---
    if isinstance(latitude, bool):
        raise ValueError(
            f"latitude phai la so huu han, khong chap nhan bool: {latitude!r}"
        )
    if isinstance(longitude, bool):
        raise ValueError(
            f"longitude phai la so huu han, khong chap nhan bool: {longitude!r}"
        )

    if not isinstance(latitude, (int, float)) or math.isnan(latitude) or math.isinf(latitude):
        raise ValueError(f"latitude phai la so huu han, nhan: {latitude!r}")
    if not (-90.0 <= latitude <= 90.0):
        raise ValueError(f"latitude phai trong [-90, 90], nhan: {latitude}")

    if not isinstance(longitude, (int, float)) or math.isnan(longitude) or math.isinf(longitude):
        raise ValueError(f"longitude phai la so huu han, nhan: {longitude!r}")
    if not (-180.0 <= longitude <= 180.0):
        raise ValueError(f"longitude phai trong [-180, 180], nhan: {longitude}")

    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) khong duoc sau end_date ({end_date})."
        )


def _to_api_date(d: date) -> str:
    """Chuyen date -> chuoi YYYYMMDD theo yeu cau cua NASA POWER API."""
    return d.strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _fetch_raw(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """Goi NASA POWER API, tra ve dict JSON thu.

    Nem:
      requests.HTTPError       - HTTP 4xx/5xx
      requests.ConnectionError - khong ket noi duoc
      requests.Timeout         - vuot timeout
      ValueError               - JSON khong hop le
    """
    params = {
        "parameters":    _PARAMETERS,
        "community":     _COMMUNITY,
        "longitude":     longitude,
        "latitude":      latitude,
        "start":         _to_api_date(start_date),
        "end":           _to_api_date(end_date),
        "format":        _FORMAT,
        "time-standard": _TIME_STD,
    }
    response = requests.get(_API_ENDPOINT, params=params, timeout=_TIMEOUT_SEC)
    response.raise_for_status()

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Phan hoi tu NASA POWER khong phai JSON hop le: {exc}. "
            f"Noi dung (200 ky tu dau): {response.text[:200]!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Parse response
# ---------------------------------------------------------------------------

def _extract_fill_value(header: dict[str, Any]) -> float:
    """Doc fill_value tu header; mac dinh -999.0 neu khong co hoac khong hop le."""
    fv = header.get("fill_value", _FILL_VALUE)
    if isinstance(fv, bool):
        return _FILL_VALUE
    try:
        return float(fv)
    except (TypeError, ValueError):
        return _FILL_VALUE


def _verify_units(parameters_meta: dict[str, Any]) -> dict[str, str]:
    """Xac minh don vi T2M va RH2M tu metadata phan hoi.

    Tra ve dict {param: unit_string} neu tat ca don vi dung.

    Nem ValueError neu:
      - Metadata don vi bi thieu hoac sai kieu (khong phai dict)
      - Metadata cua tung tham so khong phai dict hoac thieu truong units
      - Don vi nhan duoc khong khop don vi yeu cau
    """
    if not isinstance(parameters_meta, dict):
        raise ValueError(
            f"Metadata 'parameters' phai la dict, nhan {type(parameters_meta).__name__}."
        )
    units_found: dict[str, str] = {}
    for param, expected in _EXPECTED_UNITS.items():
        if param not in parameters_meta:
            raise ValueError(
                f"Metadata don vi bi thieu cho tham so '{param}'. "
                f"Yeu cau don vi '{expected}'. "
                "Khong the chap nhan du lieu khi thieu thong tin don vi."
            )
        meta = parameters_meta[param]
        if not isinstance(meta, dict):
            raise ValueError(
                f"Metadata cua tham so '{param}' phai la dict, nhan {type(meta).__name__}."
            )
        if "units" not in meta or meta["units"] is None:
            raise ValueError(
                f"Truong 'units' bi thieu trong metadata cua '{param}'. "
                f"Yeu cau don vi '{expected}'. "
                "Khong the chap nhan du lieu khi thieu thong tin don vi."
            )
        unit_str = str(meta["units"])
        if unit_str != expected:
            raise ValueError(
                f"Don vi sai cho tham so '{param}': "
                f"nhan '{unit_str}', yeu cau '{expected}'. "
                "Khong tu dong chuyen doi don vi. Kiem tra lai nguon du lieu."
            )
        units_found[param] = unit_str
    return units_found


def _is_missing(value: Any, fill_value: float) -> bool:
    """Tra ve True neu gia tri la fill_value hoac khong phai so huu han.

    Luu y: bool duoc xu ly nhu gia tri thieu (True=1.0, False=0.0 co the
    nham lan voi do am/nhiet do thuc su).
    """
    if isinstance(value, bool):
        return True
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True
    if math.isnan(v) or math.isinf(v):
        return True
    return abs(v - fill_value) <= _FILL_TOL


def _make_env_id(
    source: str,
    latitude: float,
    longitude: float,
    date_key: str,
    time_standard: str,
) -> str:
    """Tao env_id on dinh bang toan bo chuoi SHA-256 (64 ky tu hex).

    Dung hashlib.sha256 de dam bao ket qua nhat quan giua cac tien trinh
    va cac khoang truy van co ngay giao nhau.

    Khoa duoc tao tu: source, lat (chuan hoa 6 chu so thap phan), lon (6 chu so),
    date_key (YYYYMMDD) va time_standard.
    Toa do duoc lam tron 6 chu so thap phan (.6f) va chuan hoa -0.0 ve 0.0
    (thong qua phep cong + 0.0) de dam bao -0.0 va 0.0 cung bieu dien la 0.000000.
    Khong dua thoi diem tai du lieu hoac gia tri do luong vao khoa.
    Format tra ve: NASA_{date_key}_{sha256_hexdigest}
    """
    lat_val = round(float(latitude), 6) + 0.0
    lon_val = round(float(longitude), 6) + 0.0
    lat_str = f"{lat_val:.6f}"
    lon_str = f"{lon_val:.6f}"
    key_str = f"{source}|{lat_str}|{lon_str}|{date_key}|{time_standard}"
    digest = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    return f"NASA_{date_key}_{digest}"


def _eval_measurement(
    exists: bool,
    raw_val: Any,
    param: str,
    fill_value: float,
) -> tuple[float | None, list[str]]:
    """Kiem tra va danh gia mot so do (T2M hoac RH2M).

    Tra ve (valid_val_or_none, flags).
    """
    if not exists:
        return None, [f"{param}_NOT_IN_RESPONSE"]
    if raw_val is None:
        return None, [f"{param}_NULL"]
    if isinstance(raw_val, bool):
        return None, [f"{param}_BOOL_REJECTED"]
    try:
        val = float(raw_val)
    except (TypeError, ValueError):
        return None, [f"{param}_NOT_NUMERIC"]
    if math.isnan(val) or math.isinf(val):
        return None, [f"{param}_NOT_FINITE"]
    if abs(val - fill_value) <= _FILL_TOL:
        return None, [f"{param}_MISSING"]
    if param == "RH2M" and not (0.0 <= val <= 100.0):
        return None, [f"RH2M_OUT_OF_RANGE({val:.2f})"]
    return val, []


# ---------------------------------------------------------------------------
# Build DataFrame
# ---------------------------------------------------------------------------

def _build_dataframe(
    raw: dict[str, Any],
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Chuyen JSON thu thanh DataFrame chuan. Khong tu dien thieu, khong bo ngay.

    Tra ve (df, metadata_dict).
    Nem ValueError neu cau truc JSON sai kieu, thieu truong quan trong hoac don vi sai.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"Phan hoi API phai la dict JSON, nhan {type(raw).__name__}."
        )

    if "properties" not in raw:
        raise ValueError("Phan hoi API thieu truong 'properties'.")
    properties = raw["properties"]
    if not isinstance(properties, dict):
        raise ValueError(
            f"Truong 'properties' phai la dict, nhan {type(properties).__name__}."
        )

    if "parameter" not in properties:
        raise ValueError("Phan hoi API thieu 'properties.parameter'.")
    parameter_data = properties["parameter"]
    if not isinstance(parameter_data, dict):
        raise ValueError(
            f"Truong 'properties.parameter' phai la dict, nhan {type(parameter_data).__name__}."
        )

    if "header" not in raw:
        raise ValueError("Phan hoi API thieu truong 'header'.")
    header = raw["header"]
    if not isinstance(header, dict):
        raise ValueError(
            f"Truong 'header' phai la dict, nhan {type(header).__name__}."
        )

    if "parameters" not in raw:
        raise ValueError("Phan hoi API thieu truong 'parameters'.")
    params_meta = raw["parameters"]
    if not isinstance(params_meta, dict):
        raise ValueError(
            f"Truong 'parameters' phai la dict, nhan {type(params_meta).__name__}."
        )

    messages = raw.get("messages", [])
    if messages and isinstance(messages, list):
        print(f"[API messages] {messages}", file=sys.stderr)

    fill_value    = _extract_fill_value(header)
    time_standard = str(header.get("time_standard", _TIME_STD))

    # _verify_units nem ValueError neu don vi sai, thieu hoac sai kieu
    units_found   = _verify_units(params_meta)

    t2m_raw_table = parameter_data.get("T2M")
    rh2m_raw_table = parameter_data.get("RH2M")

    if t2m_raw_table is not None and not isinstance(t2m_raw_table, dict):
        raise ValueError(
            f"Bang gia tri 'parameter.T2M' phai la dict, nhan {type(t2m_raw_table).__name__}."
        )
    if rh2m_raw_table is not None and not isinstance(rh2m_raw_table, dict):
        raise ValueError(
            f"Bang gia tri 'parameter.RH2M' phai la dict, nhan {type(rh2m_raw_table).__name__}."
        )

    t2m_data: dict[str, Any] = t2m_raw_table if t2m_raw_table is not None else {}
    rh2m_data: dict[str, Any] = rh2m_raw_table if rh2m_raw_table is not None else {}

    if not t2m_data and not rh2m_data:
        raise ValueError(
            "Phan hoi API khong chua du lieu T2M hoac RH2M. "
            f"Cac key co san trong parameter: {list(parameter_data.keys())}"
        )

    # Duyet TAT CA ngay trong khoang — khong bo ngay
    records = []
    current = start_date
    n_valid   = 0
    n_missing = 0
    n_date_not_in_response = 0

    while current <= end_date:
        date_key = _to_api_date(current)
        date_iso = current.isoformat()

        has_t2m  = date_key in t2m_data
        has_rh2m = date_key in rh2m_data

        if not has_t2m and not has_rh2m:
            data_status = "DATE_NOT_IN_RESPONSE"
            n_date_not_in_response += 1
            temperature = None
            humidity = None
        else:
            raw_t2m  = t2m_data.get(date_key)
            raw_rh2m = rh2m_data.get(date_key)

            t2m_val, t2m_flags = _eval_measurement(
                exists=has_t2m,
                raw_val=raw_t2m,
                param="T2M",
                fill_value=fill_value,
            )
            rh2m_val, rh2m_flags = _eval_measurement(
                exists=has_rh2m,
                raw_val=raw_rh2m,
                param="RH2M",
                fill_value=fill_value,
            )

            all_flags = t2m_flags + rh2m_flags
            if not all_flags:
                data_status = "OK"
                n_valid += 1
                temperature = t2m_val
                humidity = rh2m_val
            else:
                data_status = "|".join(all_flags)
                n_missing += 1
                temperature = t2m_val
                humidity = rh2m_val

        env_id = _make_env_id(
            _SOURCE_NAME, latitude, longitude, date_key, time_standard
        )

        records.append({
            "env_id":        env_id,
            "latitude":      latitude,
            "longitude":     longitude,
            "date":          date_iso,
            "temperature":   temperature,
            "humidity":      humidity,
            "source":        _SOURCE_NAME,
            "time_standard": time_standard,
            "data_status":   data_status,
        })

        current += timedelta(days=1)

    df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)

    api_header = raw.get("header", {})
    api_info   = api_header.get("api", {})
    api_version = api_info.get("version", "unknown") if isinstance(api_info, dict) else str(api_info)

    metadata = {
        "endpoint":                _API_ENDPOINT,
        "parameters_requested":    _PARAMETERS,
        "community":               _COMMUNITY,
        "time_standard":           time_standard,
        "fill_value":              fill_value,
        "units":                   units_found,
        "api_version":             api_version,
        "sources":                 api_header.get("sources", []),
        "n_days_requested":        (end_date - start_date).days + 1,
        "n_valid":                 n_valid,
        "n_missing":               n_missing,
        "n_date_not_in_response":  n_date_not_in_response,
        "api_messages":            messages if isinstance(messages, list) else [],
    }

    return df, metadata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_daily_environment(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Lay du lieu nhiet do va do am hang ngay tu NASA POWER.

    Tham so:
        latitude   -- Vi do [-90, 90]
        longitude  -- Kinh do [-180, 180]
        start_date -- Ngay bat dau, dinh dang YYYY-MM-DD
        end_date   -- Ngay ket thuc, dinh dang YYYY-MM-DD

    Tra ve:
        (df, raw_json, metadata)
        df       -- DataFrame voi cot: env_id, latitude, longitude, date,
                    temperature, humidity, source, time_standard, data_status
        raw_json -- Phan hoi goc tu API (de luu nasa_power_raw.json)
        metadata -- Thong tin truy xuat nguon (de luu fetch_report.json)

    Nem:
        ValueError             -- Dau vao khong hop le hoac cau truc phan hoi la
        requests.HTTPError     -- HTTP 4xx/5xx tu API
        requests.ConnectionError -- Khong ket noi duoc
        requests.Timeout       -- Vuot timeout
    """
    d_start = _parse_date(start_date)
    d_end   = _parse_date(end_date)
    _validate_inputs(latitude, longitude, d_start, d_end)

    fetch_time = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    raw = _fetch_raw(latitude, longitude, d_start, d_end)

    df, metadata = _build_dataframe(raw, latitude, longitude, d_start, d_end)

    metadata.update({
        "fetch_utc":   fetch_time,
        "latitude":    latitude,
        "longitude":   longitude,
        "start_date":  start_date,
        "end_date":    end_date,
    })

    return df, raw, metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    """Tao thu muc dau ra. Khong ghi de — them suffix so thu tu neu ton tai."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def _versioned(name: str, suffix: str) -> Path:
        base = output_dir / f"{name}{suffix}"
        if not base.exists():
            return base
        for i in range(1, 10_000):
            candidate = output_dir / f"{name}_{i:03d}{suffix}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Khong tim duoc ten file khong bi ghi de cho {base}.")

    raw_path    = _versioned("nasa_power_raw",    ".json")
    csv_path    = _versioned("environment_daily", ".csv")
    report_path = _versioned("fetch_report",      ".json")
    return raw_path, csv_path, report_path


def main() -> None:  # pragma: no cover
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--latitude",   type=float, required=True,
                        help="Vi do (vi du: 21.028)")
    parser.add_argument("--longitude",  type=float, required=True,
                        help="Kinh do (vi du: 105.85)")
    parser.add_argument("--start",      required=True,
                        help="Ngay bat dau YYYY-MM-DD")
    parser.add_argument("--end",        required=True,
                        help="Ngay ket thuc YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Thu muc luu ket qua (mac dinh: .)")
    args = parser.parse_args()

    print(f"[NASA POWER] Truy van ({args.latitude}, {args.longitude}) "
          f"tu {args.start} den {args.end} ...")

    try:
        df, raw, metadata = fetch_daily_environment(
            latitude=args.latitude,
            longitude=args.longitude,
            start_date=args.start,
            end_date=args.end,
        )
    except ValueError as exc:
        print(f"[LOI DAU VAO] {exc}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(
            f"[LOI] Vuot qua timeout {_TIMEOUT_SEC}s. "
            "Thu lai hoac rut ngan khoang ngay.",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.exceptions.ConnectionError as exc:
        print(f"[LOI KET NOI] {exc}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        print(f"[LOI HTTP] {exc}", file=sys.stderr)
        sys.exit(1)

    raw_path, csv_path, report_path = _make_output_paths(args.output_dir)

    raw_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    report = {
        "request": {
            "latitude":   args.latitude,
            "longitude":  args.longitude,
            "start_date": args.start,
            "end_date":   args.end,
        },
        **metadata,
        "output_files": {
            "raw_json":     raw_path.name,
            "csv_table":    csv_path.name,
            "fetch_report": report_path.name,
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_req  = metadata["n_days_requested"]
    n_ok   = metadata["n_valid"]
    n_miss = metadata["n_missing"]
    n_err  = metadata["n_date_not_in_response"]
    print(f"  Ngay yeu cau           : {n_req}")
    print(f"  Hop le (OK)            : {n_ok}")
    print(f"  Thieu thong so         : {n_miss}")
    print(f"  Khong co trong phan hoi: {n_err}")
    print(f"  Da luu -> {raw_path.name}")
    print(f"         -> {csv_path.name}")
    print(f"         -> {report_path.name}")


if __name__ == "__main__":
    main()
