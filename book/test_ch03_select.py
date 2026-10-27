import unittest
import pandas as pd
import ch03_select


def table(rows):
    # ㎡単価90万円と110万円の取引を50件ずつ置き、四分位範囲を20万円に固定する。
    base = dict(area="70", built_year="2005年", year="2025", distance="6", station="用賀", name="土台")
    spread = [dict(price_yen=str(unit * 10000 * 70)) for unit in [90] * 50 + [110] * 50]
    return pd.DataFrame([dict(base, **row) for row in spread + rows], dtype=str)


def kept(rows):
    data, _ = ch03_select.select_training(table(rows))
    return sorted(data.loc[data["name"] != "土台", "name"])


class SelectTests(unittest.TestCase):
    def test_area_boundary(self):
        rows = [dict(name="50㎡", area="50", price_yen="50000000"),
                dict(name="49.9㎡", area="49.9", price_yen="49900000"),
                dict(name="空欄", area="", price_yen="70000000")]
        self.assertEqual(kept(rows), ["50㎡"])

    def test_building_year(self):
        rows = [dict(name="2005年", price_yen="70000000"), dict(name="空欄", built_year="", price_yen="70000000"),
                dict(name="取引より後", built_year="2026年", price_yen="70000000")]
        self.assertEqual(kept(rows), ["2005年"])

    def test_distance_range_is_excluded(self):
        rows = [dict(name="6分", price_yen="70000000"), dict(name="範囲", distance="30分～60分", price_yen="70000000"),
                dict(name="空欄", distance="", price_yen="70000000")]
        self.assertEqual(kept(rows), ["6分"])

    def test_outlier_boundary(self):
        # 境界は30万円と170万円。境界ちょうどは除外する。
        rows = [dict(name="上ちょうど", price_yen=str(170 * 10000 * 70)),
                dict(name="上の内側", price_yen=str(169 * 10000 * 70)),
                dict(name="下ちょうど", price_yen=str(30 * 10000 * 70)),
                dict(name="下の内側", price_yen=str(31 * 10000 * 70))]
        self.assertEqual(kept(rows), ["上の内側", "下の内側"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
