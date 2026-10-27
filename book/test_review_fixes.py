import unittest
import numpy as np
import pandas as pd
from ch04_compare import adjusted_r2
from ch04_cv import errors_on_fold


class ReviewFixTests(unittest.TestCase):
    def test_comparison_rejects_dependent_columns(self):
        with self.assertRaisesRegex(ValueError, "係数が一つ"):
            adjusted_r2(pd.DataFrame(dict(a=[1., 1., 1.], b=[1., 1., 1.])), np.array([1., 2., 3.]))

    def test_comparison_accepts_full_rank(self):
        value = adjusted_r2(pd.DataFrame(dict(a=[1., 1., 1., 1.], b=[0., 1., 2., 3.])), np.array([1., 2., 3., 4.]))
        self.assertAlmostEqual(value, 1.)

    def test_cv_rejects_dependent_columns(self):
        matrix = pd.DataFrame(dict(a=[1., 1., 1., 1.], b=[1., 1., 1., 1.]))
        data = pd.DataFrame(dict(price_yen=[100., 110., 120., 130.], area=[10., 10., 10., 10.]))
        with self.assertRaisesRegex(ValueError, "係数が一つ"):
            errors_on_fold(matrix, data, [0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
