import numpy as np
from numpy import ndarray
from tp_utils.type_utils import checked_type

from tp_maths.brownians.brownian_bridge import BrownianBridge
from tp_maths.brownians.uniform_generator import UniformGenerator


class BrownianGenerator:
    def __init__(self, uniform_generator: UniformGenerator):
        self.uniform_generator = checked_type(uniform_generator, UniformGenerator)

    def generate(self, n_variables: int, n_paths: int, times: ndarray):  # (variable, path, time)
        n_times = len(times)
        uniforms = self.uniform_generator.generate(n_paths, n_variables * n_times)
        brownians = np.zeros((n_variables, n_paths, n_times))
        bridge = BrownianBridge(times)

        for i_var in range(n_variables):
            brownians[i_var, :, :] = bridge.generate(uniforms[:, i_var::n_variables])

        return brownians
