import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import predict

MODEL = json.loads(predict.MODEL.read_text(encoding="utf-8"))
ROOM = dict(station="用賀", distance=6, area=70, built_year=2005, renovation="未改装", year=2025)


class PredictTests(unittest.TestCase):
    def test_inside(self):
        result = predict.answer(ROOM, MODEL)
        self.assertEqual(result["status"], "予測しました")
        self.assertEqual(result["predicted_total_yen"], 97116011)
        self.assertEqual(result["predicted_total"], "約9712万円")
        self.assertIn("2025年の取引価格の水準", result["meaning"])
        self.assertIn("実際の取引価格に対する割合", result["error"])
        self.assertIn("高く予測したか低く予測したかは区別していません", result["error"])

    def test_only_2025(self):
        for year in (2021, 2024, 2026, "2024"):
            with self.subTest(year=year):
                result = predict.answer(dict(ROOM, year=year), MODEL)
                self.assertEqual(result["status"], "予測できません")
                self.assertIn("2025年", result["reasons"][0])
                self.assertNotIn("predicted_total_yen", result)

    def test_outside(self):
        result = predict.answer(dict(ROOM, station="渋谷", area=300), MODEL)
        self.assertEqual(result["status"], "予測できません")
        self.assertEqual(len(result["reasons"]), 2)
        self.assertIn("予測できる範囲", result["scope"])

    def test_missing(self):
        result = predict.answer(dict(ROOM, built_year=""), MODEL)
        self.assertEqual(result["reasons"], ["入力が足りません：建築年"])

    def test_invalid_numeric_input(self):
        for field, label, value in (("area", "面積", "70㎡"), ("built_year", "建築年", "不明"),
                                    ("distance", "徒歩分数", "不明"), ("year", "取引年", "不明"),
                                    ("area", "面積", "NaN"), ("area", "面積", True)):
            with self.subTest(field=field, value=value):
                result = predict.answer(dict(ROOM, **{field: value}), MODEL)
                self.assertEqual(result["status"], "予測できません")
                self.assertTrue(any(label in reason and str(value) in reason for reason in result["reasons"]))

    def test_invalid_renovation(self):
        for value in ("", None, "不明", "改装あり"):
            with self.subTest(value=value):
                result = predict.answer(dict(ROOM, renovation=value), MODEL)
                self.assertEqual(result["status"], "予測できません")
                self.assertTrue(any("改装" in reason for reason in result["reasons"]))

    def test_numeric_strings(self):
        result = predict.answer(dict(ROOM, area="70", distance="6", built_year="2005", year="2025"), MODEL)
        self.assertEqual(result["predicted_total_yen"], 97116011)


if __name__ == "__main__":
    unittest.main(verbosity=2)
