import unittest

import numpy as np

from ch04_apply import apply_model


def model():
    return dict(reference=dict(station="用賀", year="2025"),
                columns=["intercept", "distance", "age", "area", "renovation"],
                coefficients=[np.log(1000000), -.01, -.02, .001, .1])


def property_row(**changes):
    return dict(dict(room="A", station="用賀", distance=6, area=70, built_year=2005, renovation="未改装", year=2025), **changes)


class ApplyTests(unittest.TestCase):
    def test_continuous_distance(self):
        for value in (5, 5.1, 6, 10, 10.2, 11, 15, 15.1, 16):
            result = apply_model(model(), property_row(distance=value))
            expected = np.exp(np.log(1000000) - .01 * value - .02 * 20 + .001 * 70)
            self.assertAlmostEqual(result["predicted_yen_per_m2"], expected)

    def test_continuous_age(self):
        for age in (14, 15, 24, 25, 34, 35):
            result = apply_model(model(), property_row(built_year=2025 - age))
            self.assertAlmostEqual(result["predicted_yen_per_m2"], np.exp(np.log(1000000) - .01 * 6 - .02 * age + .001 * 70))

    def test_continuous_area(self):
        for area in (50, 69.9, 70, 89.9, 90):
            result = apply_model(model(), property_row(area=area))
            expected = np.exp(np.log(1000000) - .01 * 6 - .02 * 20 + .001 * area)
            self.assertAlmostEqual(result["predicted_total_yen"], expected * area)

    def test_uses_model_column_order(self):
        fitted = model()
        order = [0, 4, 3, 1, 2]
        fitted["columns"] = [fitted["columns"][i] for i in order]
        fitted["coefficients"] = [fitted["coefficients"][i] for i in order]
        result = apply_model(fitted, property_row())
        self.assertAlmostEqual(result["predicted_total_yen"], apply_model(model(), property_row())["predicted_total_yen"], places=5)

    def test_renovation(self):
        before = apply_model(model(), property_row())["predicted_yen_per_m2"]
        after = apply_model(model(), property_row(renovation="改装済み"))["predicted_yen_per_m2"]
        self.assertAlmostEqual(after / before, np.exp(.1))

    def test_unknown_station_and_year(self):
        for changes in (dict(station="渋谷"), dict(year=2026)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                apply_model(model(), property_row(**changes))

    def test_invalid_renovation(self):
        for value in ("", "不明", None, "改装あり"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "改装"):
                apply_model(model(), property_row(renovation=value))

    def test_output_fields_only(self):
        output = apply_model(model(), property_row())
        self.assertEqual(set(output), {"room", "station", "distance", "area", "built_year", "renovation", "year",
                                       "predicted_yen_per_m2", "predicted_total_yen"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
