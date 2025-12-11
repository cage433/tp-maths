from typing import Optional

import numpy as np
from numpy import ndarray
from numpy.linalg import svd
from tp_maths.brownians.brownian_generator import BrownianGenerator
from tp_maths.brownians.uniform_generator import SOBOL_UNIFORM_GENERATOR
from tp_quantity.quantity_array import QtyArray
from tp_quantity.uom import SCALAR
from tp_utils.type_utils import checked_list_type, checked_type
from tp_quantity.quantity import UOM, Qty


class VectorPath:
    def __init__(
            self,
            uoms: list[UOM],        # (variable)
            times: ndarray,         # (time)
            path: ndarray           # (variable, path, time)
    ):
        self.uoms: list[UOM] = checked_list_type(uoms, UOM)
        self.times: ndarray = checked_type(times, ndarray)
        self.path: ndarray = checked_type(path, ndarray)

        assert path.ndim == 3, f"Expected path shape, {path.shape}, to be 3D"
        assert times.ndim == 1, f"Expected times shape, {path.shape}, to be 1D"
        self.n_variables, self.n_paths, self.n_times = path.shape
        assert len(times) == self.n_times, f"Mismatch in time lengths, {len(times)} vs {self.n_times}"
        assert len(uoms) == self.n_variables, \
            f"Mismatch in uoms {len(uoms)} vs {self.n_variables}"

    def variable_sample(self, i_variable: int, i_time: int) -> QtyArray:
        return QtyArray(self.path[i_variable, :, i_time], self.uoms[i_variable])

    @staticmethod
    def brownian_paths(
            n_variables: int,
            times: ndarray,
            n_paths: int,
            generator: Optional[BrownianGenerator] = None,
    ) -> 'VectorPath':
        generator = generator or SOBOL_UNIFORM_GENERATOR
        brownians = generator.generate(n_variables, n_paths, times)
        return VectorPath(
            [SCALAR for _ in range(brownians.shape[0])],
            times,
            brownians
        )

    def correlated(self, rho_matrix: ndarray) -> 'VectorPath':
        assert rho_matrix.ndim == 2, "Expected square rho matrix"
        assert rho_matrix.shape == (self.n_variables, self.n_variables), "Expected square rho matrix"

        U, S, _ = svd(rho_matrix)
        L = np.matmul(U, np.diag(np.sqrt(S)))
        correlated_path = np.einsum('kf,fpt->kpt', L, self.path)
        return VectorPath(self.uoms, self.times, correlated_path)

    def scaled(self, vols: ndarray) -> 'VectorPath':
        scaled_paths = np.einsum("v,vpt->vpt", vols, self.path)             # (variable, path, time)
        return VectorPath(
            self.uoms,
            self.times,
            scaled_paths
        )

    def with_drifts(self, drifts: ndarray) -> 'VectorPath':
        drift_by_time = np.broadcast_to(
            np.expand_dims(
                np.einsum("v,t -> vt", drifts, self.times),
                axis=1
            ),
            self.path.shape
        )
        return VectorPath(
            self.uoms,
            self.times,
            self.path + drift_by_time
        )

    def with_lognormal_adjustments(self, vols: ndarray) -> 'VectorPath':
        drifts = vols * vols * -0.5
        return self.with_drifts(drifts)

    def exp(self) -> 'VectorPath':
        assert all(u == SCALAR for u in self.uoms), f"UOMs must be SCALAR for call to exp()"
        return VectorPath(
            self.uoms,
            self.times,
            np.exp(self.path)
        )

    def with_prices(self, prices: list[Qty]) -> 'VectorPath':
        assert len(prices) == self.n_variables, f"Require {self.n_variables} prices, got {prices}"
        price_values = np.asarray([p.value for p in prices])
        price_uoms = [p.uom for p in prices]
        priced_path = np.einsum('f, fpt -> fpt', price_values, self.path)
        return VectorPath(
            [u1 * u2 for u1, u2 in zip(self.uoms, price_uoms)],
            self.times,
            priced_path
        )