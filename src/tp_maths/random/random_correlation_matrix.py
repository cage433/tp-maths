import numpy as np
from tp_random_tests.random_number_generator import RandomNumberGenerator


class RandomCorrelationMatrix:
    @staticmethod
    def truly_random(rng: RandomNumberGenerator, num_factors: int) -> np.ndarray:
        return RandomCorrelationMatrix._build(rng, num_factors, max_theta=np.pi)

    @staticmethod
    def with_positive_correlations(rng: RandomNumberGenerator, num_factors: int) -> np.ndarray:
        return RandomCorrelationMatrix._build(rng, num_factors, max_theta=np.pi / 4.0)

    @staticmethod
    def highly_correlated(rng: RandomNumberGenerator, num_factors: int) -> np.ndarray:
        return RandomCorrelationMatrix._build(rng, num_factors, max_theta=np.pi / 12.0)

    @staticmethod
    def _build(rng: RandomNumberGenerator, N: int, max_theta: float) -> np.ndarray:
        V = np.zeros(shape=(N, N))
        for i_factor in range(N):
            thetas = rng.uniform(size=N, x1=-max_theta, x2=max_theta)
            v = np.ones(N)
            for i in range(1, N):
                v[i] = v[i - 1] * np.sin(thetas[i - 1])
            for i in range(N - 1):
                v[i] = v[i] * np.cos(thetas[i])
            V[i_factor] = v

        # Prevent floating point errors making the result non positive-definite
        M = np.diag(np.ones(N) + 1e-9)
        for i_row in range(N):
            for i_col in range(i_row):
                rho = np.dot(V[i_row], V[i_col])
                M[i_row, i_col] = rho
                M[i_col, i_row] = rho
        return M
