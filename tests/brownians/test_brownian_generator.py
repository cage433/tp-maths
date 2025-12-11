import numpy as np
from numpy import ndarray
from scipy.stats import pearsonr
from tp_random_tests.random_number_generator import RandomNumberGenerator
from tp_random_tests.random_test_case import RandomisedTest

from tp_maths.statistics_test_case import StatisticsTestCase
from tp_maths.brownians.brownian_generator import BrownianGenerator
from tp_maths.brownians.uniform_generator import SOBOL_UNIFORM_GENERATOR, PseudoUniformGenerator


class BrownianGeneratorTest(StatisticsTestCase):

    @staticmethod
    def random_brownian_generator(rng: RandomNumberGenerator) -> BrownianGenerator:
        uniform_generator = SOBOL_UNIFORM_GENERATOR
        if rng.is_heads():
            uniform_generator = PseudoUniformGenerator(seed=rng.randint(999999))
        return BrownianGenerator(uniform_generator)

    @RandomisedTest(number_of_runs=5)
    def test_shape(self, rng):
        n_paths = 10
        n_variables = rng.randint(1, 4)
        n_times = rng.randint(10, 20)
        times = rng.random_times(n_times)

        generator = self.random_brownian_generator(rng)
        brownians = generator.generate(n_variables, n_paths, times)
        self.assertEqual(brownians.shape, (n_variables, n_paths, n_times))

    @RandomisedTest(number_of_runs=5)
    def test_independence(self, rng: RandomNumberGenerator):
        n_variables = rng.randint(2, 4)
        n_times = rng.randint(1, 4)
        times = rng.random_times(n_times)

        i_time = rng.randint(n_times)
        i_var, j_var = rng.shuffle(list(range(n_variables)))[:2]

        generator = self.random_brownian_generator(rng)

        def random_brownians(n_paths: int) -> ndarray:
            return generator.generate(n_variables, n_paths, times)

        def sample_mean(brownians: np.ndarray) -> float:
            sample_i = brownians[i_var, :, i_time]
            return float(np.std(sample_i))

        self.check_statistic(
            "Std Dev",
            random_brownians,
            sample_mean,
            tol=0.03,
            expected=np.sqrt(times[i_time])
        )

        def sample_rho(brownians: np.ndarray) -> float:
            sample_i = brownians[i_var, :, i_time]
            sample_j = brownians[j_var, :, i_time]
            return pearsonr(sample_i, sample_j)[0]

        self.check_statistic(
            "Rho",
            random_brownians,
            sample_rho,
            tol=0.03,
            expected=0.0,
        )
