from abc import ABC, abstractmethod

import numpy as np

from tp_maths.brownians.sobol_generator import SobolGenerator


class UniformGenerator(ABC):
    @abstractmethod
    def generate(self, n_paths: int, n_variables: int) -> np.ndarray: #(path, variable)
        pass


class PseudoUniformGenerator(UniformGenerator):
    def __init__(self, seed: int):
        self.seed = seed

    def generate(self, n_paths: int, n_variables: int) -> np.ndarray:
        random_state = np.random.RandomState(self.seed)
        return random_state.uniform(size=(n_paths, n_variables))


class SobolUniformGenerator(UniformGenerator):
    def generate(self, n_paths: int, n_variables: int) -> np.ndarray:
        sobol = SobolGenerator.cached(n_variables)
        return sobol.generate(n_paths)

SOBOL_UNIFORM_GENERATOR = SobolUniformGenerator()