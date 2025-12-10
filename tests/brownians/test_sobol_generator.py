from unittest import TestCase

import numpy as np
from scipy.stats._qmc import Sobol

from scipy.stats import pearsonr
from tp_maths.brownians.sobol_generator import SobolGenerator
from tp_random_tests.random_test_case import RandomisedTest


class SobolGeneratorTest(TestCase):

    @RandomisedTest(number_of_runs=3)
    def test_distribution_is_uniform_like(self, rng):

        n_variables = rng.randint(1, 10)
        sd = SobolGenerator(n_variables)
        n_paths = 1 << rng.randint(8, 12)
        sample = sd.generate(n_paths)

        i_var = rng.randint(n_variables)
        v = sample[:, i_var]

        se = np.std(v) / np.sqrt(n_paths)
        self.assertAlmostEqual(np.mean(v), 0.5, delta=4.0 * se)
        self.assertAlmostEqual(np.std(v), 1.0 / np.sqrt(12), delta=0.01)

        if n_variables >= 2:
            j_var = rng.choice([i for i in range(n_variables) if i != i_var])
            self.assertAlmostEqual(
                pearsonr(sample[:, i_var], sample[:, j_var])[0],
                0.0,
                delta=0.03
            )

    def test_matches_scipy(self):
        n_variables = 2000
        sd = SobolGenerator(n_variables)
        n_paths = 1 << 10
        print(n_paths)
        sample1 = sd.generate(n_paths)
        sample2 = Sobol(n_variables, scramble=False, bits=52).random(n_paths * 2)[1:n_paths + 1]
        for i in range(n_paths):
            diff = sample1[i, :] - sample2[i, :]
            self.assertAlmostEqual(np.mean(diff), 0.0, delta=0.01)
            self.assertAlmostEqual(np.max(diff), 0.0, delta=0.01)
            self.assertAlmostEqual(np.min(diff), 0.0, delta=0.01)
