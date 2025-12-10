from unittest import TestCase

import numpy as np
from tp_maths.brownians.brownian_bridge import BrownianBridge
from tp_random_tests.random_test_case import RandomisedTest


class BrownianBridgeTests(TestCase):

    @RandomisedTest(number_of_runs=10, num_allowed_failures=1)
    def test_mean_and_std_dev(self, rng):
        n_times = np.random.randint(1, 10)
        times = rng.random_times(n_times)
        n_paths = 1024 << 3
        uniforms = rng.uniform(size=(n_paths, n_times))
        bldr = BrownianBridge(times)
        paths = bldr.generate(uniforms)

        for i_time in range(n_times):
            sample = paths[:, i_time]
            expected_std_dev = np.sqrt(times[i_time])
            self.assertAlmostEqual(
                sample.mean(),
                0.0,
                delta=expected_std_dev * 4.0 / np.sqrt(n_paths),
            )

            sample_std_dev = sample.std()
            self.assertAlmostEqual(
                sample_std_dev,
                expected_std_dev,
                delta=0.02,
            )
