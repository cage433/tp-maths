import unittest
from numbers import Number
from typing import TypeVar, Callable, Optional, Union

from tp_quantity.quantity import Qty
from tp_quantity.uom import SCALAR


class StatisticsTestCase(unittest.TestCase):
    S = TypeVar("S")

    def _tolerance(
            self,
            samples: S,
            tol: Optional[Union[float, Qty]],
            tol_func: Optional[Callable[[S], Union[float, Qty]]] = None,
    ) -> Qty:
        tol_ = tol or tol_func(samples)
        return Qty.to_qty(tol_)

    def check_statistic(self,
                        msg: str,
                        sample_generator: Callable[[int], S],
                        statistic_func: Callable[[S], Union[float, Qty]],
                        expected: Union[float, Union[float, Qty]],
                        tol: Optional[Union[float, Qty]] = None,
                        tol_func: Optional[Callable[[S], Union[float, Qty]]] = None,
                        init_n_samples=2_000,
                        log_on_try: int = 6,
                        n_tries: int = 6):
        has_passed = False
        i_try = 0
        n_samples = init_n_samples
        observed = 0
        tol_ = 0
        if isinstance(expected, Number):
            expected = Qty(expected, SCALAR)
        while not has_passed and i_try < n_tries:
            samples = sample_generator(n_samples)
            observed = Qty.to_qty(statistic_func(samples))
            tol_ = self._tolerance(samples, tol, tol_func)
            error = (observed - expected)
            if error.abs < tol_:
                has_passed = True
            if i_try >= log_on_try:
                print(
                    f"i:{i_try}, N:{n_samples} O:{observed:1.3f}, EXP:{expected:1.3f}, ERR:{error:1.3f}, TOL:{tol:1.3f}")
            i_try += 1
            n_samples *= 2
        if not has_passed:
            self.fail(
                f"{msg} failed: Expected {expected:1.3f} (+/- {tol_}), last observed after {n_tries} tries was {observed:1.3f}")
