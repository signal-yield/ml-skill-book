import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import ch03_select
import ch05_evaluate as evaluation


def rows():
    return pd.DataFrame([dict(station="用賀", distance="6", area="70", built_year="2005年", year="2025",
                              renovation="未改装", price_yen="70000000"),
                         dict(station="用賀", distance="7", area="70", built_year="2000年", year="2025",
                              renovation="不明", price_yen="90000000")])


class EvaluationTests(unittest.TestCase):
    def test_basic_selection_does_not_read_price(self):
        data = rows()
        data["price_yen"] = "価格を使わない"
        with patch.object(pd.Series, "quantile", side_effect=AssertionError("四分位を計算しない")):
            selected, flow = ch03_select.select_basic(data)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(flow), 4)

    def test_test_loader_called_once_and_rule5_skipped(self):
        train = rows().iloc[:1].copy()
        test = rows()
        test.loc[0, "price_yen"] = "1000000000000"
        model = dict(columns=["intercept", "distance", "age", "area", "renovation"], coefficients=[np.log(1000000), 0., 0., 0., 0.])
        with patch.object(evaluation, "load_training_csv", return_value=train), \
             patch.object(evaluation, "load_test_csv", return_value=test) as loader, \
             patch.object(ch03_select, "select_training", return_value=(train, [])), \
             patch.object(evaluation, "fit_model", return_value=model), \
             patch.object(pd.Series, "quantile", side_effect=AssertionError("テストで四分位を計算しない")):
            result = evaluation.evaluate("synthetic.csv")
        loader.assert_called_once_with("synthetic.csv")
        self.assertFalse(result["test_rule5_applied"])
        self.assertEqual(result["test_flow"][-1], ["改装が分かる", 1])
        self.assertEqual(result["model_on_test"]["count"], 1)
        self.assertGreater(result["model_on_test"]["median"], .99)

    def test_absolute_percentage_error_and_linear_percentile(self):
        data = pd.DataFrame(dict(area=[1., 1., 1., 1.], price_yen=[100., 100., 100., 100.]))
        result = evaluation.errors(np.array([90., 120., 70., 140.]), data)
        self.assertAlmostEqual(result["median"], .25)
        self.assertAlmostEqual(result["p90"], .37)


if __name__ == "__main__":
    unittest.main(verbosity=2)
