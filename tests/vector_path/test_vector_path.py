from functools import cached_property
from typing import Optional, Callable

import numpy as np
from numpy import ndarray
from scipy.stats import pearsonr
from tp_maths.random.random_correlation_matrix import RandomCorrelationMatrix
from tp_maths.statistics_test_case import StatisticsTestCase
from tp_maths.vector_path.vector_path import VectorPath
from tp_quantity.quantity import Qty
from tp_quantity.uom import MWH, EUR, MT, USD
from tp_random_tests.random_number_generator import RandomNumberGenerator
from tp_random_tests.random_test_case import RandomisedTest

from tests.brownians.test_brownian_generator import BrownianGeneratorTest


class VectorPathTestCase(StatisticsTestCase):
    class RandomScenario:
        def __init__(self, rng: RandomNumberGenerator, n_variables: Optional[int] = None,
                     n_times: Optional[int] = None):
            self.rng = rng
            self.n_variables = n_variables or rng.randint(2, 4)
            self.n_times = n_times or rng.randint(1, 20)
            self.i_time = rng.randint(self.n_times)

            self.i_var = rng.randint(self.n_variables)
            self.times = rng.random_times(self.n_times)

        @cached_property
        def j_var(self):
            if self.n_variables == 1:
                raise ValueError("Not enough variables")
            return self.rng.choice([v for v in range(self.n_variables) if v != self.i_var])

    def _brownian_paths(self, rng: RandomNumberGenerator, scenario: RandomScenario, n_paths: int) -> VectorPath:
        times = scenario.times
        generator = rng.maybe(BrownianGeneratorTest.random_uniform_generator(rng))
        return VectorPath.brownian_paths(
            scenario.n_variables,
            times,
            n_paths,
            generator
        )

    def _sample_std_dev(self, i_var: int, i_time: int) -> Callable[[VectorPath], float]:
        def sd(vp: VectorPath) -> float:
            return vp.variable_sample(i_var, i_time).std

        return sd

    def _sample_mean(self, i_var: int, i_time: int) -> Callable[[VectorPath], Qty]:
        def sm(vp: VectorPath):
            return vp.variable_sample(i_var, i_time).mean

        return sm

    def _sample_historic_vol(self, i_var: int) -> Callable[[VectorPath], Qty]:
        def shv(vp: VectorPath):
            vols = np.asarray([
                vp.observed_vol(i_var, i_path)
                for i_path in range(vp.n_paths)
            ])
            return vols.mean()

        return shv

    def _sample_mean_tol(self, i_var: int, i_time: int, num_se: float) -> Callable[[VectorPath], Qty]:
        def sm(vp: VectorPath):
            sample = vp.variable_sample(i_var, i_time)
            se = sample.std_err
            return (se * Qty.to_qty(num_se)).max(Qty(0.0001, sample.uom))

        return sm

    def _sample_rho(self, i_var: int, j_var: int, i_time: int) -> Callable[
        [VectorPath], Qty]:
        def sr(vp: VectorPath):
            sample_i = vp.variable_sample(i_var, i_time).checked_scalar_values
            sample_j = vp.variable_sample(j_var, i_time).checked_scalar_values
            return pearsonr(sample_i, sample_j)[0]

        return sr

    @RandomisedTest(number_of_runs=10)
    def test_from_brownians(self, rng):
        scenario = self.RandomScenario(rng)
        i_time, i_var, j_var, times = scenario.i_time, scenario.i_var, scenario.j_var, scenario.times

        def random_vector_path(n_paths: int) -> VectorPath:
            return self._brownian_paths(rng, scenario, n_paths)

        self.check_statistic(
            "Mean",
            random_vector_path,
            self._sample_mean(i_var, i_time),
            tol_func=self._sample_mean_tol(i_var, i_time, num_se=3),
            expected=0.0,
        )

        self.check_statistic(
            "Std dev",
            random_vector_path,
            self._sample_std_dev(i_var, i_time),
            tol=0.01,
            expected=np.sqrt(times[i_time]),
        )

        self.check_statistic(
            "Rho",
            random_vector_path,
            self._sample_rho(i_var, j_var, i_time),
            tol=0.01,
            expected=0.0,
        )

    @RandomisedTest(number_of_runs=5)
    def test_correlation(self, rng: RandomNumberGenerator):
        scenario = self.RandomScenario(rng)
        i_time, i_var, j_var, n_variables = scenario.i_time, scenario.i_var, scenario.j_var, scenario.n_variables

        rho_matrix = RandomCorrelationMatrix.truly_random(rng, n_variables)

        def random_vector_path(n_paths: int) -> VectorPath:
            return self._brownian_paths(rng, scenario, n_paths).correlated(rho_matrix)

        self.check_statistic(
            "Rho",
            random_vector_path,
            self._sample_rho(i_var, j_var, i_time),
            tol=0.01,
            expected=rho_matrix[i_var][j_var],
        )

    @RandomisedTest(number_of_runs=10)
    def test_scaling(self, rng):
        scenario = self.RandomScenario(rng)
        i_time, i_var, times, n_variables = scenario.i_time, scenario.i_var, scenario.times, scenario.n_variables

        rho_matrix = RandomCorrelationMatrix.truly_random(rng, n_variables)
        vols = np.asarray([rng.uniform() for _ in range(n_variables)])

        def random_vector_path(n_paths: int) -> VectorPath:
            return self._brownian_paths(rng, scenario, n_paths).correlated(rho_matrix).scaled(vols)

        self.check_statistic(
            "Std Dev",
            random_vector_path,
            self._sample_std_dev(i_var, i_time),
            tol=0.01,
            expected=vols[i_var] * np.sqrt(times[i_time])
        )

    # noinspection PyTypeChecker
    @RandomisedTest(number_of_runs=10)
    def test_drift(self, rng):
        scenario = self.RandomScenario(rng)
        i_time, i_var, j_var, times, n_variables = scenario.i_time, scenario.i_var, scenario.j_var, scenario.times, scenario.n_variables

        drifts: ndarray = np.asarray([rng.uniform() for _ in range(n_variables)])

        def random_vector_path(n_paths: int) -> VectorPath:
            return self._brownian_paths(rng, scenario, n_paths).with_drifts(drifts)

        self.check_statistic(
            "Mean",
            random_vector_path,
            self._sample_mean(i_var, i_time),
            tol_func=self._sample_mean_tol(i_var, i_time, num_se=3),
            expected=drifts[i_var] * times[i_time],
        )

        self.check_statistic(
            "Std dev",
            random_vector_path,
            self._sample_std_dev(i_var, i_time),
            tol=0.01,
            expected=np.sqrt(times[i_time]),
        )

    @RandomisedTest(number_of_runs=10)
    def test_exp(self, rng):
        scenario = self.RandomScenario(rng)
        i_time, i_var, j_var, times, n_variables = scenario.i_time, scenario.i_var, scenario.j_var, scenario.times, scenario.n_variables
        vols: ndarray = np.asarray([rng.uniform() * 0.5 for _ in range(n_variables)])
        rho_matrix = RandomCorrelationMatrix.truly_random(rng, n_variables)

        def random_vector_path(n_paths: int) -> VectorPath:
            return self._brownian_paths(rng, scenario, n_paths).correlated(rho_matrix).scaled(
                vols).with_lognormal_adjustments(vols).exp()

        self.check_statistic(
            "Mean",
            random_vector_path,
            self._sample_mean(i_var, i_time),
            tol_func=self._sample_mean_tol(i_var, i_time, num_se=1),
            expected=1.0,
        )

    @RandomisedTest(number_of_runs=10)
    def test_with_prices(self, rng):
        scenario = self.RandomScenario(rng)
        i_time, i_var, j_var, times, n_variables = scenario.i_time, scenario.i_var, scenario.j_var, scenario.times, scenario.n_variables
        vols: ndarray = np.asarray([rng.uniform() * 0.5 for _ in range(n_variables)])
        rho_matrix = RandomCorrelationMatrix.truly_random(rng, n_variables)
        prices = [Qty(rng.uniform(), rng.choice(MWH, EUR / MWH, MT, USD / MT)) for _ in range(n_variables)]

        def random_vector_path(n_paths: int) -> VectorPath:
            return self._brownian_paths(rng, scenario, n_paths).correlated(rho_matrix).scaled(
                vols).with_lognormal_adjustments(vols).exp().with_prices(prices)

        self.check_statistic(
            "Mean",
            random_vector_path,
            self._sample_mean(i_var, i_time),
            tol_func=self._sample_mean_tol(i_var, i_time, num_se=1),
            expected=prices[i_var],
        )

    @RandomisedTest(number_of_runs=10)
    def test_historic_vol(self, rng):
        scenario = self.RandomScenario(rng, n_variables=1, n_times=100)
        vol = rng.uniform() * 0.5
        F = Qty(rng.uniform(), USD / MT)
        vols: ndarray = np.asarray([vol])
        prices = [F]

        def random_vector_path(n_paths: int) -> VectorPath:
            brownian_paths = self._brownian_paths(rng, scenario, n_paths)
            return brownian_paths.scaled(
                vols).with_lognormal_adjustments(vols).exp().with_prices(prices)

        self.check_statistic(
            "Historic vol",
            random_vector_path,
            self._sample_historic_vol(i_var=0),
            tol=0.01,
            expected=vol,
            init_n_samples=1_000,
        )
