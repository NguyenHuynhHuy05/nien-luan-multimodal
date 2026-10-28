"""Test suite toan dien cho nasa_power_service.py.

Tat ca test su dung du lieu gia lap (mock) - khong goi mang hay NASA API that.

Luu y: Pandas tu dong chuyen None -> np.float64(nan) trong cot so.
Vi vay kiem tra gia tri thieu dung pd.isna() hoac math.isnan() thay vi assertIsNone.

Pham vi kiem tra:
  - _parse_date: dinh dang hop le, sai dinh dang, chuoi rong
  - _validate_inputs: toa do, khoang ngay, tu choi bool truoc khi goi mang
  - _to_api_date: chuyen doi ngay sang YYYYMMDD
  - _is_missing: fill_value, inf, nan, None, gia tri hop le, bool
  - _extract_fill_value: co/khong co fill_value, kieu sai, bool
  - _verify_units: dung, sai don vi, Kelvin, thieu tung param, thieu metadata, sai kieu
  - _make_env_id:
      + Hoi quy va cham SHA-256 (lat=-79.7835 vs -62.8555 cung lon=105.85)
      + Chuan hoa -0.0 va 0.0
      + Toa do int 21 va float 21.0
      + On dinh giua cac tien trinh (subprocess voi PYTHONHASHSEED khac nhau)
      + On dinh giua cac khoang truy van co ngay giao nhau
  - _build_dataframe:
      + Kiem tra kieu cau truc JSON truoc khi truy cap (raw, properties, parameter, header, parameters)
      + Ngay vang hoan toan -> DATE_NOT_IN_RESPONSE
      + Ngay co hai gia tri null -> T2M_NULL|RH2M_NULL
      + Chi mot thong so bi thieu (giu thong so con lai)
      + Nhieu loi dong thoi (vd T2M=True & RH2M=-999 -> T2M_BOOL_REJECTED|RH2M_MISSING)
      + Thong ke: n_valid + n_missing + n_date_not_in_response == n_days_requested == len(df)
      + Khoang ngay di qua cuoi thang va ngay nhuan (2024-02-29)
  - fetch_daily_environment:
      + requests mock thanh cong, timeout, loi ket noi, loi HTTP, JSON khong hop le
      + Toa do boolean bi tu choi truoc khi goi mang
  - _make_output_paths: versioning khong ghi de
  - Luong CLI (main):
      + Tao du 3 file (raw JSON, CSV, fetch_report JSON)
      + Doc lai doi chieu du lieu, cot va so dem
      + Chay lan 2 tao suffix _001 khong ghi de
      + output_files trong bao cao tro dung file tuong ung
      + Phan hoi loi thi CLI ket thuc bang SystemExit(1)
"""
from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# Them src/data vao sys.path de import module
_SRC_DATA = Path(__file__).parent
if str(_SRC_DATA) not in sys.path:
    sys.path.insert(0, str(_SRC_DATA))

import nasa_power_service as svc  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: Phan hoi gia lap chuan
# ---------------------------------------------------------------------------

def _make_valid_raw(
    t2m_override: dict | None = None,
    rh2m_override: dict | None = None,
    fill_value: float = -999.0,
    units_override: dict | None = None,
) -> dict:
    """Tao phan hoi JSON gia lap hop le cho 2024-01-01 -> 2024-01-03."""
    t2m  = {"20240101": 20.95, "20240102": 21.47, "20240103": 16.11}
    rh2m = {"20240101": 88.32, "20240102": 88.22, "20240103": 82.67}
    if t2m_override is not None:
        t2m = t2m_override
    if rh2m_override is not None:
        rh2m = rh2m_override

    default_units = {
        "T2M":  {"units": "C",  "longname": "Temperature at 2 Meters"},
        "RH2M": {"units": "%",  "longname": "Relative Humidity at 2 Meters"},
    }
    if units_override is not None:
        default_units.update(units_override)

    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [105.85, 21.028, 87.73]},
        "properties": {
            "parameter": {"T2M": t2m, "RH2M": rh2m}
        },
        "header": {
            "title": "NASA/POWER Source Native Resolution Daily Data",
            "api": {"version": "v2.10.0", "name": "POWER Daily API"},
            "sources": ["MERRA2", "POWER"],
            "fill_value": fill_value,
            "time_standard": "LST",
            "start": "20240101",
            "end": "20240103",
        },
        "messages": [],
        "parameters": default_units,
        "times": {"data": 0.192, "process": 0.01},
    }


def _mock_response(raw: dict | str, status_code: int = 200):
    """Tao mock requests.Response."""
    import requests
    resp = MagicMock()
    resp.status_code = status_code
    if isinstance(raw, dict):
        resp.json.return_value = raw
        resp.text = json.dumps(raw)
    else:
        resp.text = raw
        resp.json.side_effect = json.JSONDecodeError("err", "", 0)

    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} Error", response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _is_na(val) -> bool:
    """Kiem tra gia tri thieu: None hoac NaN (pandas chuyen None -> NaN)."""
    return pd.isna(val)


# ---------------------------------------------------------------------------
# 1. _parse_date
# ---------------------------------------------------------------------------

class TestParseDate(unittest.TestCase):

    def test_valid_date(self):
        d = svc._parse_date("2024-01-15")
        self.assertEqual(d, date(2024, 1, 15))

    def test_valid_date_strip_whitespace(self):
        d = svc._parse_date("  2024-03-31  ")
        self.assertEqual(d, date(2024, 3, 31))

    def test_wrong_format_slash(self):
        with self.assertRaises(ValueError):
            svc._parse_date("15/01/2024")

    def test_wrong_format_yyyymmdd(self):
        with self.assertRaises(ValueError):
            svc._parse_date("20240115")

    def test_invalid_date(self):
        with self.assertRaises(ValueError):
            svc._parse_date("2024-02-30")

    def test_empty_string(self):
        with self.assertRaises(ValueError):
            svc._parse_date("")


# ---------------------------------------------------------------------------
# 2. _validate_inputs
# ---------------------------------------------------------------------------

class TestValidateInputs(unittest.TestCase):

    def _ok(self, lat, lon, s, e):
        svc._validate_inputs(lat, lon, date.fromisoformat(s), date.fromisoformat(e))

    def test_valid(self):
        self._ok(21.028, 105.85, "2024-01-01", "2024-01-31")

    def test_same_start_end(self):
        self._ok(0.0, 0.0, "2024-06-01", "2024-06-01")

    def test_lat_too_high(self):
        with self.assertRaises(ValueError):
            self._ok(91.0, 105.0, "2024-01-01", "2024-01-02")

    def test_lat_too_low(self):
        with self.assertRaises(ValueError):
            self._ok(-90.1, 0.0, "2024-01-01", "2024-01-02")

    def test_lon_too_high(self):
        with self.assertRaises(ValueError):
            self._ok(10.0, 181.0, "2024-01-01", "2024-01-02")

    def test_lon_too_low(self):
        with self.assertRaises(ValueError):
            self._ok(10.0, -180.1, "2024-01-01", "2024-01-02")

    def test_start_after_end(self):
        with self.assertRaises(ValueError):
            self._ok(10.0, 100.0, "2024-01-10", "2024-01-05")

    def test_lat_nan(self):
        with self.assertRaises(ValueError):
            self._ok(float("nan"), 100.0, "2024-01-01", "2024-01-02")

    def test_lat_inf(self):
        with self.assertRaises(ValueError):
            self._ok(float("inf"), 100.0, "2024-01-01", "2024-01-02")

    # --- Boolean coordinates ---

    def test_lat_bool_true_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._ok(True, 105.85, "2024-01-01", "2024-01-02")
        self.assertIn("bool", str(ctx.exception).lower())

    def test_lat_bool_false_raises(self):
        with self.assertRaises(ValueError):
            self._ok(False, 105.85, "2024-01-01", "2024-01-02")

    def test_lon_bool_true_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._ok(21.028, True, "2024-01-01", "2024-01-02")
        self.assertIn("bool", str(ctx.exception).lower())

    def test_lon_bool_false_raises(self):
        with self.assertRaises(ValueError):
            self._ok(21.028, False, "2024-01-01", "2024-01-02")


# ---------------------------------------------------------------------------
# 3. _to_api_date
# ---------------------------------------------------------------------------

class TestToApiDate(unittest.TestCase):

    def test_format(self):
        self.assertEqual(svc._to_api_date(date(2024, 1, 5)), "20240105")

    def test_leading_zeros(self):
        self.assertEqual(svc._to_api_date(date(2023, 9, 3)), "20230903")


# ---------------------------------------------------------------------------
# 4. _is_missing
# ---------------------------------------------------------------------------

class TestIsMissing(unittest.TestCase):

    FV = -999.0

    def test_exact_fill_value(self):
        self.assertTrue(svc._is_missing(-999.0, self.FV))

    def test_near_fill_value_within_tol(self):
        self.assertTrue(svc._is_missing(-999.4, self.FV))

    def test_normal_value(self):
        self.assertFalse(svc._is_missing(25.5, self.FV))

    def test_zero(self):
        self.assertFalse(svc._is_missing(0.0, self.FV))

    def test_none(self):
        self.assertTrue(svc._is_missing(None, self.FV))

    def test_nan(self):
        self.assertTrue(svc._is_missing(float("nan"), self.FV))

    def test_inf(self):
        self.assertTrue(svc._is_missing(float("inf"), self.FV))

    def test_string_number(self):
        self.assertFalse(svc._is_missing("25.5", self.FV))

    def test_non_numeric_string(self):
        self.assertTrue(svc._is_missing("abc", self.FV))

    def test_bool_true_is_missing(self):
        self.assertTrue(svc._is_missing(True, self.FV))

    def test_bool_false_is_missing(self):
        self.assertTrue(svc._is_missing(False, self.FV))


# ---------------------------------------------------------------------------
# 5. _extract_fill_value
# ---------------------------------------------------------------------------

class TestExtractFillValue(unittest.TestCase):

    def test_present(self):
        self.assertEqual(svc._extract_fill_value({"fill_value": -9999.0}), -9999.0)

    def test_absent(self):
        self.assertEqual(svc._extract_fill_value({}), svc._FILL_VALUE)

    def test_bad_type(self):
        self.assertEqual(svc._extract_fill_value({"fill_value": "bad"}), svc._FILL_VALUE)

    def test_bool_rejected(self):
        self.assertEqual(svc._extract_fill_value({"fill_value": True}), svc._FILL_VALUE)
        self.assertEqual(svc._extract_fill_value({"fill_value": False}), svc._FILL_VALUE)


# ---------------------------------------------------------------------------
# 6. _verify_units
# ---------------------------------------------------------------------------

class TestVerifyUnits(unittest.TestCase):

    def test_correct_units(self):
        meta = {
            "T2M":  {"units": "C",  "longname": "Temperature at 2 Meters"},
            "RH2M": {"units": "%",  "longname": "Relative Humidity at 2 Meters"},
        }
        result = svc._verify_units(meta)
        self.assertEqual(result["T2M"], "C")
        self.assertEqual(result["RH2M"], "%")

    def test_wrong_unit_t2m_kelvin_raises(self):
        meta = {"T2M": {"units": "K"}, "RH2M": {"units": "%"}}
        with self.assertRaises(ValueError) as ctx:
            svc._verify_units(meta)
        err = str(ctx.exception)
        self.assertIn("T2M", err)
        self.assertIn("K", err)
        self.assertIn("C", err)

    def test_wrong_unit_rh2m_raises(self):
        meta = {"T2M": {"units": "C"}, "RH2M": {"units": "fraction"}}
        with self.assertRaises(ValueError) as ctx:
            svc._verify_units(meta)
        err = str(ctx.exception)
        self.assertIn("RH2M", err)
        self.assertIn("fraction", err)
        self.assertIn("%", err)

    def test_missing_t2m_param_raises(self):
        meta = {"RH2M": {"units": "%"}}
        with self.assertRaises(ValueError) as ctx:
            svc._verify_units(meta)
        self.assertIn("T2M", str(ctx.exception))

    def test_missing_rh2m_param_raises(self):
        meta = {"T2M": {"units": "C"}}
        with self.assertRaises(ValueError) as ctx:
            svc._verify_units(meta)
        self.assertIn("RH2M", str(ctx.exception))

    def test_empty_metadata_raises(self):
        with self.assertRaises(ValueError):
            svc._verify_units({})

    def test_metadata_not_dict_raises(self):
        with self.assertRaises(ValueError) as ctx:
            svc._verify_units(["not", "dict"])
        self.assertIn("dict", str(ctx.exception))

    def test_param_entry_not_dict_raises(self):
        with self.assertRaises(ValueError) as ctx:
            svc._verify_units({"T2M": "not_dict", "RH2M": {"units": "%"}})
        self.assertIn("T2M", str(ctx.exception))

    def test_missing_units_field_raises(self):
        meta = {
            "T2M":  {"longname": "Temperature at 2 Meters"},
            "RH2M": {"units": "%"},
        }
        with self.assertRaises(ValueError) as ctx:
            svc._verify_units(meta)
        self.assertIn("T2M", str(ctx.exception))


# ---------------------------------------------------------------------------
# 7. _make_env_id (Hoi quy va cham & Chuan hoa)
# ---------------------------------------------------------------------------

class TestMakeEnvId(unittest.TestCase):

    SOURCE = "NASA_POWER"
    TIME_STD = "LST"

    def _make(self, lat, lon, date_key):
        return svc._make_env_id(self.SOURCE, lat, lon, date_key, self.TIME_STD)

    def test_format_full_sha256(self):
        eid = self._make(21.028, 105.85, "20240101")
        self.assertTrue(eid.startswith("NASA_20240101_"))
        hash_part = eid[len("NASA_20240101_"):]
        self.assertEqual(len(hash_part), 64)
        int(hash_part, 16)  # hex hop le

    def test_known_collision_regression(self):
        """Hoi quy: 2 toa do nay tung bi trung 8 ky tu hash dau (311e8c69).
        Voi full 64-char sha256, chung phai tao 2 env_id khac nhau hoan toan."""
        id1 = self._make(-79.7835, 105.85, "20240101")
        id2 = self._make(-62.8555, 105.85, "20240101")
        # 8 ky tu dau tung trung
        self.assertEqual(id1[14:22], "311e8c69")
        self.assertEqual(id2[14:22], "311e8c69")
        # Nhung full ID phai khac nhau
        self.assertNotEqual(id1, id2)

    def test_negative_zero_normalization(self):
        """-0.0 va 0.0 phai tao ra cung env_id."""
        id_pos = self._make(0.0, 105.85, "20240101")
        id_neg = self._make(-0.0, 105.85, "20240101")
        self.assertEqual(id_pos, id_neg)

        id_lon_pos = self._make(21.028, 0.0, "20240101")
        id_lon_neg = self._make(21.028, -0.0, "20240101")
        self.assertEqual(id_lon_pos, id_lon_neg)

    def test_equivalent_coordinate_types(self):
        """int 21 va float 21.0 phai tao ra cung env_id."""
        id_int = self._make(21, 105, "20240101")
        id_float = self._make(21.0, 105.0, "20240101")
        self.assertEqual(id_int, id_float)

    def test_overlapping_date_queries_yield_same_id(self):
        """Cung ngay trong 2 khoang truy van giao nhau phai co cung env_id."""
        id_q1 = self._make(21.028, 105.85, "20240102")
        id_q2 = self._make(21.028, 105.85, "20240102")
        self.assertEqual(id_q1, id_q2)

    def test_different_dates_different_id(self):
        id1 = self._make(21.028, 105.85, "20240101")
        id2 = self._make(21.028, 105.85, "20240102")
        self.assertNotEqual(id1, id2)

    def test_stable_across_processes_subprocess(self):
        """env_id phai nhat quan giua cac tien trinh voi PYTHONHASHSEED khac nhau."""
        script = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, r'{str(_SRC_DATA)}')
            import nasa_power_service as svc
            eid = svc._make_env_id('NASA_POWER', 21.028, 105.85, '20240101', 'LST')
            print(eid)
        """)
        results = []
        for seed in ["0", "1", "42", "999999"]:
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = seed
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, env=env, timeout=15,
            )
            self.assertEqual(proc.returncode, 0,
                             msg=f"Subprocess failed (seed={seed}): {proc.stderr}")
            results.append(proc.stdout.strip())

        self.assertEqual(len(set(results)), 1,
                         msg=f"env_id khac nhau giua cac tien trinh: {results}")


# ---------------------------------------------------------------------------
# 8. _build_dataframe & JSON structure validation
# ---------------------------------------------------------------------------

class TestBuildDataframe(unittest.TestCase):

    START = date(2024, 1, 1)
    END   = date(2024, 1, 3)
    LAT   = 21.028
    LON   = 105.85

    def _build(self, raw, start=None, end=None):
        s = start or self.START
        e = end or self.END
        return svc._build_dataframe(raw, self.LAT, self.LON, s, e)

    # --- Kiem tra cau truc JSON truoc khi truy cap ---

    def test_raw_not_dict_raises(self):
        for bad in [[], "string", 123]:
            with self.assertRaises(ValueError) as ctx:
                self._build(bad)
            self.assertIn("dict", str(ctx.exception))

    def test_missing_properties_raises(self):
        raw = {"header": {}, "parameters": {}}
        with self.assertRaises(ValueError) as ctx:
            self._build(raw)
        self.assertIn("properties", str(ctx.exception))

    def test_properties_not_dict_raises(self):
        raw = {"properties": "not_dict", "header": {}, "parameters": {}}
        with self.assertRaises(ValueError) as ctx:
            self._build(raw)
        self.assertIn("properties", str(ctx.exception))

    def test_missing_parameter_raises(self):
        raw = {"properties": {}, "header": {}, "parameters": {}}
        with self.assertRaises(ValueError) as ctx:
            self._build(raw)
        self.assertIn("parameter", str(ctx.exception))

    def test_parameter_not_dict_raises(self):
        raw = {"properties": {"parameter": []}, "header": {}, "parameters": {}}
        with self.assertRaises(ValueError) as ctx:
            self._build(raw)
        self.assertIn("properties.parameter", str(ctx.exception))

    def test_missing_header_raises(self):
        raw = {"properties": {"parameter": {}}, "parameters": {}}
        with self.assertRaises(ValueError) as ctx:
            self._build(raw)
        self.assertIn("header", str(ctx.exception))

    def test_header_not_dict_raises(self):
        raw = {"properties": {"parameter": {}}, "header": None, "parameters": {}}
        with self.assertRaises(ValueError) as ctx:
            self._build(raw)
        self.assertIn("header", str(ctx.exception))

    def test_missing_parameters_raises(self):
        raw = {"properties": {"parameter": {}}, "header": {}}
        with self.assertRaises(ValueError) as ctx:
            self._build(raw)
        self.assertIn("parameters", str(ctx.exception))

    def test_parameters_not_dict_raises(self):
        raw = {"properties": {"parameter": {}}, "header": {}, "parameters": "bad"}
        with self.assertRaises(ValueError) as ctx:
            self._build(raw)
        self.assertIn("parameters", str(ctx.exception))

    def test_t2m_table_not_dict_raises(self):
        raw = _make_valid_raw()
        raw["properties"]["parameter"]["T2M"] = [20.95, 21.47]
        with self.assertRaises(ValueError) as ctx:
            self._build(raw)
        self.assertIn("parameter.T2M", str(ctx.exception))

    def test_rh2m_table_not_dict_raises(self):
        raw = _make_valid_raw()
        raw["properties"]["parameter"]["RH2M"] = "not_a_dict"
        with self.assertRaises(ValueError) as ctx:
            self._build(raw)
        self.assertIn("parameter.RH2M", str(ctx.exception))

    def test_empty_parameter_data_raises(self):
        raw = _make_valid_raw()
        raw["properties"]["parameter"] = {}
        with self.assertRaises(ValueError) as ctx:
            self._build(raw)
        self.assertIn("T2M", str(ctx.exception))

    # --- Phan loai ngay thieu & trang thai tung thong so ---

    def test_date_not_in_response_only_when_both_missing(self):
        """DATE_NOT_IN_RESPONSE chi khi ngay vang mat hoan toan o ca 2."""
        t2m  = {"20240101": 20.95}
        rh2m = {"20240101": 88.32}
        raw = _make_valid_raw(t2m_override=t2m, rh2m_override=rh2m)
        df, meta = self._build(raw)
        self.assertEqual(df.iloc[0]["data_status"], "OK")
        self.assertEqual(df.iloc[1]["data_status"], "DATE_NOT_IN_RESPONSE")
        self.assertEqual(df.iloc[2]["data_status"], "DATE_NOT_IN_RESPONSE")
        self.assertEqual(meta["n_valid"], 1)
        self.assertEqual(meta["n_date_not_in_response"], 2)
        self.assertEqual(meta["n_missing"], 0)

    def test_two_null_values_not_date_not_in_response(self):
        """Ngay co 2 gia tri null khong phai DATE_NOT_IN_RESPONSE ma la T2M_NULL|RH2M_NULL."""
        t2m  = {"20240101": None,  "20240102": 21.47, "20240103": 16.11}
        rh2m = {"20240101": None,  "20240102": 88.22, "20240103": 82.67}
        raw = _make_valid_raw(t2m_override=t2m, rh2m_override=rh2m)
        df, meta = self._build(raw)
        row0 = df.iloc[0]
        self.assertEqual(row0["data_status"], "T2M_NULL|RH2M_NULL")
        self.assertTrue(_is_na(row0["temperature"]))
        self.assertTrue(_is_na(row0["humidity"]))
        self.assertEqual(meta["n_missing"], 1)
        self.assertEqual(meta["n_valid"], 2)
        self.assertEqual(meta["n_date_not_in_response"], 0)

    def test_only_one_parameter_missing_preserves_other(self):
        """T2M missing nhung RH2M hop le -> giu RH2M."""
        raw = _make_valid_raw(
            t2m_override={"20240101": -999.0, "20240102": 21.47, "20240103": 16.11}
        )
        df, meta = self._build(raw)
        row0 = df.iloc[0]
        self.assertEqual(row0["data_status"], "T2M_MISSING")
        self.assertTrue(_is_na(row0["temperature"]))
        self.assertAlmostEqual(row0["humidity"], 88.32)
        self.assertEqual(meta["n_missing"], 1)
        self.assertEqual(meta["n_valid"], 2)

    def test_date_missing_in_one_param_only(self):
        """Ngay co trong RH2M nhung khong co trong T2M."""
        t2m  = {"20240102": 21.47, "20240103": 16.11}  # khong co 20240101
        rh2m = {"20240101": 88.32, "20240102": 88.22, "20240103": 82.67}
        raw = _make_valid_raw(t2m_override=t2m, rh2m_override=rh2m)
        df, meta = self._build(raw)
        row0 = df.iloc[0]
        self.assertEqual(row0["data_status"], "T2M_NOT_IN_RESPONSE")
        self.assertTrue(_is_na(row0["temperature"]))
        self.assertAlmostEqual(row0["humidity"], 88.32)
        self.assertEqual(meta["n_missing"], 1)

    def test_multiple_errors_aggregated(self):
        """Tong hop day du loi khi ca hai thong so deu loi."""
        # T2M=True (bool), RH2M=-999 (missing)
        t2m  = {"20240101": True, "20240102": 21.47, "20240103": 16.11}
        rh2m = {"20240101": -999.0, "20240102": 88.22, "20240103": 82.67}
        raw = _make_valid_raw(t2m_override=t2m, rh2m_override=rh2m)
        df, meta = self._build(raw)
        row0 = df.iloc[0]
        self.assertEqual(row0["data_status"], "T2M_BOOL_REJECTED|RH2M_MISSING")
        self.assertTrue(_is_na(row0["temperature"]))
        self.assertTrue(_is_na(row0["humidity"]))
        self.assertEqual(meta["n_missing"], 1)

    def test_non_numeric_and_out_of_range_aggregated(self):
        """T2M='abc' va RH2M=150.0."""
        t2m  = {"20240101": "abc", "20240102": 21.47, "20240103": 16.11}
        rh2m = {"20240101": 150.0, "20240102": 88.22, "20240103": 82.67}
        raw = _make_valid_raw(t2m_override=t2m, rh2m_override=rh2m)
        df, _ = self._build(raw)
        self.assertEqual(df.iloc[0]["data_status"], "T2M_NOT_NUMERIC|RH2M_OUT_OF_RANGE(150.00)")

    def test_non_finite_values(self):
        """nan va inf gay NOT_FINITE."""
        t2m  = {"20240101": float("nan"), "20240102": 21.47, "20240103": 16.11}
        rh2m = {"20240101": float("inf"), "20240102": 88.22, "20240103": 82.67}
        raw = _make_valid_raw(t2m_override=t2m, rh2m_override=rh2m)
        df, _ = self._build(raw)
        self.assertEqual(df.iloc[0]["data_status"], "T2M_NOT_FINITE|RH2M_NOT_FINITE")

    def test_status_counts_sum_to_total_rows(self):
        """Tong n_valid + n_missing + n_date_not_in_response phai bang n_days_requested va bang len(df)."""
        t2m  = {"20240101": 20.95, "20240102": -999.0}  # 20240103 vang mat
        rh2m = {"20240101": 88.32, "20240102": 88.22}
        raw = _make_valid_raw(t2m_override=t2m, rh2m_override=rh2m)
        df, meta = self._build(raw)
        self.assertEqual(meta["n_valid"], 1)
        self.assertEqual(meta["n_missing"], 1)
        self.assertEqual(meta["n_date_not_in_response"], 1)
        self.assertEqual(meta["n_valid"] + meta["n_missing"] + meta["n_date_not_in_response"], 3)
        self.assertEqual(len(df), 3)

    def test_month_transition_and_leap_year(self):
        """Kiem tra khoang ngay di qua cuoi thang va ngay nhuan (2024-02-29)."""
        t2m = {
            "20240228": 22.0,
            "20240229": 23.5,
            "20240301": 24.0,
        }
        rh2m = {
            "20240228": 80.0,
            "20240229": 85.0,
            "20240301": 78.0,
        }
        raw = _make_valid_raw(t2m_override=t2m, rh2m_override=rh2m)
        raw["header"]["start"] = "20240228"
        raw["header"]["end"] = "20240301"
        start = date(2024, 2, 28)
        end = date(2024, 3, 1)
        df, meta = self._build(raw, start=start, end=end)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.iloc[1]["date"], "2024-02-29")
        self.assertEqual(meta["n_valid"], 3)


# ---------------------------------------------------------------------------
# 9. fetch_daily_environment (requests mock)
# ---------------------------------------------------------------------------

class TestFetchDailyEnvironment(unittest.TestCase):

    @patch("nasa_power_service.requests.get")
    def test_success(self, mock_get):
        raw = _make_valid_raw()
        mock_get.return_value = _mock_response(raw)
        df, returned_raw, meta = svc.fetch_daily_environment(
            21.028, 105.85, "2024-01-01", "2024-01-03"
        )
        self.assertEqual(len(df), 3)
        self.assertTrue((df["data_status"] == "OK").all())
        self.assertEqual(returned_raw, raw)
        self.assertEqual(meta["n_valid"], 3)

    @patch("nasa_power_service.requests.get")
    def test_metadata_has_fetch_utc(self, mock_get):
        mock_get.return_value = _mock_response(_make_valid_raw())
        _, _, meta = svc.fetch_daily_environment(
            21.028, 105.85, "2024-01-01", "2024-01-01"
        )
        self.assertIn("fetch_utc", meta)
        self.assertTrue(meta["fetch_utc"].endswith("Z"))

    @patch("nasa_power_service.requests.get")
    def test_http_error(self, mock_get):
        import requests as req
        mock_get.return_value = _mock_response({}, status_code=404)
        with self.assertRaises(req.exceptions.HTTPError):
            svc.fetch_daily_environment(21.028, 105.85, "2024-01-01", "2024-01-03")

    @patch("nasa_power_service.requests.get")
    def test_connection_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("timeout")
        with self.assertRaises(req.exceptions.ConnectionError):
            svc.fetch_daily_environment(21.028, 105.85, "2024-01-01", "2024-01-03")

    @patch("nasa_power_service.requests.get")
    def test_timeout(self, mock_get):
        import requests as req
        mock_get.side_effect = req.exceptions.Timeout()
        with self.assertRaises(req.exceptions.Timeout):
            svc.fetch_daily_environment(21.028, 105.85, "2024-01-01", "2024-01-03")

    @patch("nasa_power_service.requests.get")
    def test_invalid_json(self, mock_get):
        mock_get.return_value = _mock_response("not json", status_code=200)
        with self.assertRaises(ValueError) as ctx:
            svc.fetch_daily_environment(21.028, 105.85, "2024-01-01", "2024-01-03")
        self.assertIn("JSON", str(ctx.exception))

    def test_invalid_latitude(self):
        with self.assertRaises(ValueError):
            svc.fetch_daily_environment(200.0, 105.85, "2024-01-01", "2024-01-03")

    def test_invalid_longitude(self):
        with self.assertRaises(ValueError):
            svc.fetch_daily_environment(21.0, 200.0, "2024-01-01", "2024-01-03")

    def test_lat_bool_raises_before_network(self):
        with patch("nasa_power_service.requests.get") as mock_get:
            with self.assertRaises(ValueError):
                svc.fetch_daily_environment(True, 105.85, "2024-01-01", "2024-01-03")
            mock_get.assert_not_called()

    def test_lon_bool_raises_before_network(self):
        with patch("nasa_power_service.requests.get") as mock_get:
            with self.assertRaises(ValueError):
                svc.fetch_daily_environment(21.028, False, "2024-01-01", "2024-01-03")
            mock_get.assert_not_called()

    @patch("nasa_power_service.requests.get")
    def test_wrong_units_in_response_raises(self, mock_get):
        raw = _make_valid_raw(units_override={"T2M": {"units": "K"}})
        mock_get.return_value = _mock_response(raw)
        with self.assertRaises(ValueError) as ctx:
            svc.fetch_daily_environment(21.028, 105.85, "2024-01-01", "2024-01-03")
        self.assertIn("T2M", str(ctx.exception))


# ---------------------------------------------------------------------------
# 10. _make_output_paths & CLI integration
# ---------------------------------------------------------------------------

class TestCLIAndOutputPaths(unittest.TestCase):

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "new_subdir"
            self.assertFalse(out_dir.exists())
            svc._make_output_paths(out_dir)
            self.assertTrue(out_dir.exists())

    def test_no_overwrite_versioning(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            (out_dir / "nasa_power_raw.json").write_text("{}")
            (out_dir / "environment_daily.csv").write_text("a,b")
            (out_dir / "fetch_report.json").write_text("{}")
            raw, csv, rpt = svc._make_output_paths(out_dir)
            self.assertEqual(raw.name, "nasa_power_raw_001.json")
            self.assertEqual(csv.name, "environment_daily_001.csv")
            self.assertEqual(rpt.name, "fetch_report_001.json")

    @patch("nasa_power_service.requests.get")
    def test_cli_full_workflow_and_no_overwrite(self, mock_get):
        """Kiem tra toan bo luong CLI luu vao thu muc tam va khong ghi de khi chay lai."""
        raw_mock = _make_valid_raw()
        mock_get.return_value = _mock_response(raw_mock)

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            test_argv = [
                "nasa_power_service.py",
                "--latitude", "21.028",
                "--longitude", "105.85",
                "--start", "2024-01-01",
                "--end", "2024-01-03",
                "--output-dir", str(out_dir),
            ]

            # Chay lan 1
            with patch("sys.argv", test_argv):
                svc.main()

            raw_file = out_dir / "nasa_power_raw.json"
            csv_file = out_dir / "environment_daily.csv"
            rpt_file = out_dir / "fetch_report.json"

            self.assertTrue(raw_file.exists())
            self.assertTrue(csv_file.exists())
            self.assertTrue(rpt_file.exists())

            # Doc lai doi chieu du lieu lan 1
            df1 = pd.read_csv(csv_file, encoding="utf-8-sig")
            self.assertEqual(len(df1), 3)
            self.assertEqual(list(df1.columns), svc.OUTPUT_COLUMNS)
            self.assertEqual(df1.iloc[0]["date"], "2024-01-01")
            self.assertAlmostEqual(df1.iloc[0]["temperature"], 20.95)

            rpt1 = json.loads(rpt_file.read_text(encoding="utf-8"))
            self.assertEqual(rpt1["output_files"]["raw_json"], "nasa_power_raw.json")
            self.assertEqual(rpt1["output_files"]["csv_table"], "environment_daily.csv")
            self.assertEqual(rpt1["output_files"]["fetch_report"], "fetch_report.json")
            self.assertEqual(rpt1["n_valid"], 3)
            self.assertEqual(rpt1["n_missing"], 0)

            # Chay lan 2 voi cung output_dir -> khong duoc ghi de
            with patch("sys.argv", test_argv):
                svc.main()

            raw_file_001 = out_dir / "nasa_power_raw_001.json"
            csv_file_001 = out_dir / "environment_daily_001.csv"
            rpt_file_001 = out_dir / "fetch_report_001.json"

            self.assertTrue(raw_file_001.exists())
            self.assertTrue(csv_file_001.exists())
            self.assertTrue(rpt_file_001.exists())

            # output_files cua lan 2 tro dung vao cac file _001
            rpt2 = json.loads(rpt_file_001.read_text(encoding="utf-8"))
            self.assertEqual(rpt2["output_files"]["raw_json"], "nasa_power_raw_001.json")
            self.assertEqual(rpt2["output_files"]["csv_table"], "environment_daily_001.csv")
            self.assertEqual(rpt2["output_files"]["fetch_report"], "fetch_report_001.json")

    @patch("nasa_power_service.requests.get")
    def test_cli_invalid_response_exits_with_error(self, mock_get):
        """Phan hoi API khong hop le phai ket thuc bang SystemExit(1), khong bao thanh cong."""
        # Tra ve JSON thieu parameters
        raw_bad = {"properties": {"parameter": {}}, "header": {}}
        mock_get.return_value = _mock_response(raw_bad)

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            test_argv = [
                "nasa_power_service.py",
                "--latitude", "21.028",
                "--longitude", "105.85",
                "--start", "2024-01-01",
                "--end", "2024-01-03",
                "--output-dir", str(out_dir),
            ]
            with patch("sys.argv", test_argv):
                with self.assertRaises(SystemExit) as ctx:
                    svc.main()
                self.assertEqual(ctx.exception.code, 1)

            # Khong co file csv nao duoc tao khi loi xay ra
            csv_file = out_dir / "environment_daily.csv"
            self.assertFalse(csv_file.exists())


# ---------------------------------------------------------------------------
# 11. DataFrame Integrity Integration
# ---------------------------------------------------------------------------

class TestDataFrameIntegrity(unittest.TestCase):

    @patch("nasa_power_service.requests.get")
    def test_latitude_longitude_preserved(self, mock_get):
        mock_get.return_value = _mock_response(_make_valid_raw())
        df, _, _ = svc.fetch_daily_environment(21.028, 105.85, "2024-01-01", "2024-01-03")
        self.assertTrue((df["latitude"] == 21.028).all())
        self.assertTrue((df["longitude"] == 105.85).all())

    @patch("nasa_power_service.requests.get")
    def test_valid_temperature_is_finite(self, mock_get):
        mock_get.return_value = _mock_response(_make_valid_raw())
        df, _, _ = svc.fetch_daily_environment(21.028, 105.85, "2024-01-01", "2024-01-03")
        ok_rows = df[df["data_status"] == "OK"]
        for _, row in ok_rows.iterrows():
            self.assertTrue(math.isfinite(float(row["temperature"])))
            self.assertTrue(math.isfinite(float(row["humidity"])))

    @patch("nasa_power_service.requests.get")
    def test_valid_humidity_in_range(self, mock_get):
        mock_get.return_value = _mock_response(_make_valid_raw())
        df, _, _ = svc.fetch_daily_environment(21.028, 105.85, "2024-01-01", "2024-01-03")
        ok_rows = df[df["data_status"] == "OK"]
        self.assertTrue((ok_rows["humidity"] >= 0).all())
        self.assertTrue((ok_rows["humidity"] <= 100).all())

    @patch("nasa_power_service.requests.get")
    def test_no_crop_or_season_columns(self, mock_get):
        mock_get.return_value = _mock_response(_make_valid_raw())
        df, _, _ = svc.fetch_daily_environment(21.028, 105.85, "2024-01-01", "2024-01-03")
        self.assertNotIn("crop",   df.columns)
        self.assertNotIn("season", df.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
