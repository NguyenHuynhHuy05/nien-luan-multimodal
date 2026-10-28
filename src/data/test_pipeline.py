"""Kiểm tra pipeline ảnh–văn bản và pipeline môi trường.

Các DataFrame fixture trong file này chỉ phục vụ kiểm tra tự động.
Không xuất thành dữ liệu nghiên cứu.
"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from generate_dataset import load_inventory, split_inventory, synthesize
from pipeline_utils import validate_dataset
from preprocessing import preprocess
from text_bank import PROFILES


def fixture():
    """Tạo DataFrame giả chỉ dùng cho test; không phải dữ liệu nghiên cứu."""
    rows = []
    for label in sorted(PROFILES):
        for i in range(20):
            for copy in range(2):
                rows.append({"image_path": f"{label}/{i}-{copy}.jpg", "label": label,
                             "group_id": f"{label}:{i}"})
    return pd.DataFrame(rows)


class ImageTextPipelineTests(unittest.TestCase):
    def setUp(self):
        self.df, _ = synthesize(split_inventory(fixture(), 42), 42)

    def test_no_environment_columns(self):
        """Pipeline ảnh–văn bản không tạo cột môi trường."""
        env_cols = {"temperature", "humidity", "season", "environment_source"}
        self.assertFalse(env_cols & set(self.df.columns),
                         "Bảng ảnh–văn bản không nên có cột môi trường")

    def test_expected_columns(self):
        expected = {"image_path", "label", "group_id", "split",
                    "description", "text_level", "text_family_id", "text_source"}
        self.assertEqual(expected, set(self.df.columns))

    def test_duplicate_bytes_are_grouped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for label in PROFILES:
                folder = root / label
                folder.mkdir()
                for i in range(8):
                    # Bytes giả chỉ để kiểm tra hash, không dùng cho train ảnh.
                    (folder / f"{i}.jpg").write_bytes(f"{label}:{i}".encode())
                (folder / "copy.jpg").write_bytes((folder / "0.jpg").read_bytes())
            inv, mode = load_inventory(data_dir=root)
            self.assertEqual(mode, "sha256_image_bytes")
            self.assertEqual(len(inv) - inv.group_id.nunique(), 12)
            split = split_inventory(inv, 42)
            self.assertEqual(split.groupby("group_id").split.nunique().max(), 1)

    def test_generation_reproducible_and_held_out_families(self):
        again, _ = synthesize(split_inventory(fixture(), 42), 42)
        pd.testing.assert_frame_equal(self.df, again)
        validate_dataset(self.df)
        self.assertEqual(self.df.groupby("group_id").split.nunique().max(), 1)

    def test_reject_overlap_after_cleaning(self):
        bad = self.df.copy()
        ti = bad.index[bad.split.eq("train")][0]
        vi = bad.index[bad.split.eq("test")][0]
        bad.loc[vi, "description"] = bad.loc[ti, "description"].upper() + " !!!"
        with self.assertRaisesRegex(ValueError, "trùng"):
            validate_dataset(bad)

    def test_preprocessing_keeps_split(self):
        """Preprocessing giữ nguyên split và chỉ fit label encoder trên Train."""
        parts, labels, _ = preprocess(self.df.copy())
        for split, part in parts.items():
            self.assertEqual(set(part.image_path),
                             set(self.df.loc[self.df.split.eq(split), "image_path"]))
        # Kiểm tra label_id nhất quán
        self.assertTrue(all("label_id" in part.columns for part in parts.values()))
        self.assertTrue(all("clean_description" in part.columns for part in parts.values()))

    def test_preprocessing_drops_env_columns(self):
        """Preprocessing loại cột môi trường cũ nếu đầu vào có sẵn chúng."""
        df_with_env = self.df.copy()
        df_with_env["temperature"] = 25.0
        df_with_env["humidity"] = 60.0
        df_with_env["season"] = "Mùa Hạ"
        df_with_env["environment_source"] = "synthetic"
        parts, _, _ = preprocess(df_with_env)
        for part in parts.values():
            self.assertFalse({"temperature", "humidity", "season", "environment_source"} & set(part.columns))

    def test_reject_without_split(self):
        with self.assertRaisesRegex(ValueError, "Thiếu cột"):
            preprocess(self.df.drop(columns="split"))

    def test_synthesize_drops_leftover_env_columns(self):
        """synthesize() giới hạn đầu ra đúng 8 cột và không giữ lại cột môi trường."""
        from generate_dataset import IMAGE_TEXT_COLUMNS
        inv = split_inventory(fixture(), 42)
        inv["temperature"] = 28.5
        inv["humidity"] = 70.0
        inv["season"] = "Mùa Mưa"
        inv["environment_source"] = "sensor_net"
        inv["extra_random_column"] = "random"
        out_df, _ = synthesize(inv, 42)
        expected = ["image_path", "label", "group_id", "split",
                    "description", "text_level", "text_family_id", "text_source"]
        self.assertEqual(list(out_df.columns), expected)
        self.assertEqual(IMAGE_TEXT_COLUMNS, expected)
        self.assertNotIn("temperature", out_df.columns)
        self.assertNotIn("humidity", out_df.columns)
        self.assertNotIn("season", out_df.columns)
        self.assertNotIn("environment_source", out_df.columns)
        self.assertNotIn("extra_random_column", out_df.columns)


class EnvironmentPipelineTests(unittest.TestCase):
    """Kiểm tra pipeline môi trường độc lập, không phụ thuộc bảng ảnh–văn bản."""

    def _make_env_fixture(self):
        """Tạo DataFrame môi trường nhỏ chỉ dùng cho test; không phải dữ liệu nghiên cứu."""
        return pd.DataFrame({
            "env_id": ["e001", "e002", "e003", "e004", "e005",
                       "e006", "e007", "e008", "e009", "e010"],
            "crop": ["tomato"] * 5 + ["corn"] * 5,
            "temperature": [22.5, 28.0, 15.3, 30.1, 25.0,
                            20.0, 33.5, 18.2, 26.7, 24.0],
            "humidity": [65.0, 78.0, 55.0, 82.0, 70.0,
                         60.0, 88.0, 50.0, 75.0, 68.0],
            "source": ["test_fixture"] * 10,
            "split": ["train"] * 6 + ["val"] * 2 + ["test"] * 2,
        })

    def test_env_no_image_path_or_description(self):
        """Bảng môi trường không yêu cầu image_path hoặc description."""
        from preprocessing_environment import validate_env_data
        df = self._make_env_fixture()
        self.assertNotIn("image_path", df.columns)
        self.assertNotIn("description", df.columns)
        errors = validate_env_data(df)
        self.assertEqual(len(errors), 0, f"Lỗi validation: {errors}")

    def test_env_fit_train_only(self):
        """Scaler môi trường chỉ fit trên split Train."""
        from preprocessing_environment import preprocess_env
        df = self._make_env_fixture()
        result = preprocess_env(df)
        train_data = df[df["split"] == "train"]
        np.testing.assert_allclose(result["scaler"].data_min_,
                                   train_data[["temperature", "humidity"]].min().values)
        np.testing.assert_allclose(result["scaler"].data_max_,
                                   train_data[["temperature", "humidity"]].max().values)

    def test_env_reject_invalid_humidity(self):
        from preprocessing_environment import validate_env_data
        df = self._make_env_fixture()
        df.loc[0, "humidity"] = 150.0
        errors = validate_env_data(df)
        self.assertTrue(any("humidity" in e.lower() or "0" in e or "100" in e for e in errors),
                        f"Phải báo lỗi humidity ngoài [0,100]: {errors}")

    def test_env_reject_nan(self):
        from preprocessing_environment import validate_env_data
        df = self._make_env_fixture()
        df.loc[1, "temperature"] = float("nan")
        errors = validate_env_data(df)
        self.assertTrue(len(errors) > 0, "Phải báo lỗi giá trị NaN")

    def test_env_season_optional(self):
        """Nếu không có cột season, không lỗi; nếu có thì mã hóa."""
        from preprocessing_environment import validate_env_data, preprocess_env
        df = self._make_env_fixture()
        errors = validate_env_data(df)
        self.assertEqual(len(errors), 0)
        # Thêm season
        df["season"] = ["Mùa Xuân", "Mùa Hạ", "Mùa Thu", "Mùa Đông", "Mùa Xuân",
                        "Mùa Hạ", "Mùa Thu", "Mùa Đông", "Mùa Xuân", "Mùa Hạ"]
        result = preprocess_env(df)
        self.assertIn("season_encoder", result)

    def test_env_no_split_raises(self):
        """Pipeline môi trường cần split để biết fit trên đâu."""
        from preprocessing_environment import preprocess_env
        df = self._make_env_fixture().drop(columns="split")
        with self.assertRaisesRegex(ValueError, "split"):
            preprocess_env(df)

    def test_env_reject_missing_or_whitespace_or_invalid_split(self):
        """Từ chối preprocessing nếu thiếu split, split chỉ có khoảng trắng, hoặc ngoài train/val/test."""
        from preprocessing_environment import preprocess_env, validate_env_data
        # 1. Thiếu split (NaN) ở 1 dòng
        df_na = self._make_env_fixture()
        df_na.loc[0, "split"] = np.nan
        self.assertTrue(len(validate_env_data(df_na)) > 0)
        with self.assertRaises(ValueError):
            preprocess_env(df_na)

        # 2. Split chỉ có khoảng trắng
        df_ws = self._make_env_fixture()
        df_ws.loc[0, "split"] = "   "
        self.assertTrue(len(validate_env_data(df_ws)) > 0)
        with self.assertRaises(ValueError):
            preprocess_env(df_ws)

        # 3. Split ngoài train/val/test
        df_bad = self._make_env_fixture()
        df_bad.loc[0, "split"] = "eval"
        self.assertTrue(len(validate_env_data(df_bad)) > 0)
        with self.assertRaises(ValueError):
            preprocess_env(df_bad)

    def test_env_output_rows_equal_input_rows(self):
        """Tổng số dòng đầu ra bằng tổng số dòng đầu vào, không âm thầm bỏ dòng."""
        from preprocessing_environment import preprocess_env
        df = self._make_env_fixture()
        result = preprocess_env(df)
        total_out = sum(len(p) for p in result["parts"].values())
        self.assertEqual(total_out, len(df))
        self.assertEqual(result["report"]["total_output_rows"], len(df))
        self.assertEqual(result["report"]["rows"], len(df))

    def test_env_reject_whitespace_and_missing_env_id_crop_source(self):
        """Kiểm tra giá trị thiếu và chuỗi rỗng sau khi strip cho env_id, crop, source."""
        from preprocessing_environment import validate_env_data, preprocess_env
        for col in ["env_id", "crop", "source"]:
            # Giá trị thiếu (NaN)
            df_nan = self._make_env_fixture()
            df_nan.loc[0, col] = np.nan
            errors = validate_env_data(df_nan)
            self.assertTrue(any(col in e for e in errors), f"Phải bắt lỗi NaN ở cột {col}")
            with self.assertRaises(ValueError):
                preprocess_env(df_nan)

            # Chuỗi rỗng sau khi strip khoảng trắng
            df_ws = self._make_env_fixture()
            df_ws.loc[0, col] = "   "
            errors = validate_env_data(df_ws)
            self.assertTrue(any(col in e for e in errors), f"Phải bắt lỗi khoảng trắng ở cột {col}")
            with self.assertRaises(ValueError):
                preprocess_env(df_ws)

    def test_env_reject_duplicate_env_id_after_strip(self):
        """Kiểm tra env_id trùng lặp sau khi chuẩn hóa khoảng trắng."""
        from preprocessing_environment import validate_env_data, preprocess_env
        df = self._make_env_fixture()
        df.loc[1, "env_id"] = "  e001  "  # Dòng 0 là "e001"
        errors = validate_env_data(df)
        self.assertTrue(any("trùng lặp" in e and "env_id" in e for e in errors))
        with self.assertRaises(ValueError):
            preprocess_env(df)

    def test_env_season_all_train_missing_raises(self):
        """Nếu season được cung cấp nhưng toàn bộ Train thiếu mùa, báo lỗi rõ ràng trước khi fit encoder."""
        from preprocessing_environment import preprocess_env
        df = self._make_env_fixture()
        # Train có 6 dòng đầu, gán thiếu; Val/Test có 4 dòng sau, gán có mùa
        df["season"] = [np.nan] * 6 + ["Mùa Xuân", "Mùa Hạ", "Mùa Thu", "Mùa Đông"]
        with self.assertRaises(ValueError) as ctx:
            preprocess_env(df)
        err_msg = str(ctx.exception).lower()
        self.assertTrue("season" in err_msg or "mùa" in err_msg or "train" in err_msg)

    def test_env_season_partial_missing_raises(self):
        """Nếu chỉ một số dòng thiếu mùa trong khi những dòng khác có, báo lỗi để người dùng xử lý, không tự điền."""
        from preprocessing_environment import validate_env_data, preprocess_env
        df = self._make_env_fixture()
        df["season"] = ["Mùa Xuân"] * 9 + [np.nan]
        errors = validate_env_data(df)
        self.assertTrue(any("season" in e or "mùa" in e for e in errors))
        with self.assertRaises(ValueError):
            preprocess_env(df)

        # Trường hợp rỗng sau strip
        df_ws = self._make_env_fixture()
        df_ws["season"] = ["Mùa Xuân"] * 9 + ["   "]
        errors = validate_env_data(df_ws)
        self.assertTrue(any("season" in e or "mùa" in e for e in errors))
        with self.assertRaises(ValueError):
            preprocess_env(df_ws)

    def test_env_season_all_rows_missing_skips_encoding(self):
        """Nếu không có thông tin mùa ở bất kỳ dòng nào, bỏ qua mã hóa mùa và ghi rõ trong báo cáo, không tạo season_nan."""
        from preprocessing_environment import preprocess_env
        # 1. Cột season toàn NaN
        df_nan = self._make_env_fixture()
        df_nan["season"] = [np.nan] * len(df_nan)
        result = preprocess_env(df_nan)
        self.assertIsNone(result["season_encoder"])
        self.assertFalse(result["report"]["has_season"])
        self.assertIn("season_note", result["report"])
        for part in result["parts"].values():
            self.assertFalse(any(c.startswith("season_") for c in part.columns),
                             "Không được tạo cột season_nan hay season_None")

        # 2. Cột season toàn chuỗi rỗng / khoảng trắng
        df_ws = self._make_env_fixture()
        df_ws["season"] = ["   "] * len(df_ws)
        result_ws = preprocess_env(df_ws)
        self.assertIsNone(result_ws["season_encoder"])
        self.assertFalse(result_ws["report"]["has_season"])
        for part in result_ws["parts"].values():
            self.assertFalse(any(c.startswith("season_") for c in part.columns))


if __name__ == "__main__":
    unittest.main()
