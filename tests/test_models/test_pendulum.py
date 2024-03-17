import math
from typing import List

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_almost_equal_nulp

from hamilflow.models.pendulum import Pendulum, PendulumIC, PendulumSystem

_LIS_OMEGA0 = [0.3, 0.6, 1.5]
_LIS_THETA0 = [-1.5, -0.3, 0.2, 0.4, +1.4]
_LIS_LIS_TIMES = [[-5.5, 0.0, 0.5, 1.0]]


class TestPendulumSystem:
    @pytest.mark.parametrize("omega0", [-1.0, 0])
    def test_omega0_range(self, omega0: float) -> None:
        with pytest.raises(ValueError):
            _ = PendulumSystem(omega0=omega0)

    def test_init(self) -> None:
        _ = PendulumSystem(1)


class TestPendulumIC:
    @pytest.mark.parametrize("theta0", [-2.0, 2.0])
    def test_theta0_range(self, theta0: float) -> None:
        with pytest.raises(ValueError):
            _ = PendulumIC(theta0=theta0)

    def test_init(self) -> None:
        _ = PendulumIC(1)

    @pytest.mark.parametrize("theta0", _LIS_THETA0)
    def test_k(self, theta0: float) -> None:
        ic = PendulumIC(theta0=theta0)
        assert ic.k == np.sin(ic.theta0 / 2)


class TestPendulum:
    @pytest.mark.parametrize(
        ("omega0", "theta0", "period"),
        [(2 * math.pi, math.pi / 3, 1.0731820071493643751)],
    )
    def test_period_static(self, omega0: float, theta0: float, period: float) -> None:
        p = Pendulum(omega0, theta0)
        assert pytest.approx(p.period) == period

    @pytest.mark.parametrize("omega0", _LIS_OMEGA0)
    @pytest.mark.parametrize("theta0", _LIS_THETA0)
    @pytest.mark.parametrize("times", _LIS_LIS_TIMES)
    def test_transf(self, omega0: float, theta0: float, times: List[float]) -> None:
        p = Pendulum(omega0, theta0)
        arr_times = np.asarray(times)

        sin_u = np.sin(p.u(arr_times))
        theta_terms = np.sin(p.theta(arr_times) / 2) / p._k
        assert_array_almost_equal(theta_terms, sin_u)
        # assert_array_almost_equal_nulp(theta_terms, sin_u, 32)

    @pytest.mark.parametrize("omega0", _LIS_OMEGA0)
    @pytest.mark.parametrize("theta0", _LIS_THETA0)
    @pytest.mark.parametrize("times", _LIS_LIS_TIMES)
    def test_period_dynamic_theta(
        self, omega0: float, theta0: float, times: List[float]
    ) -> None:
        p = Pendulum(omega0, theta0)
        arr_times = np.asarray(times)
        arr_times_1 = arr_times + p.period

        theta, theta_1 = p.theta(arr_times), p.theta(arr_times_1)

        assert_array_almost_equal(theta, theta_1)
        # assert_array_almost_equal_nulp(theta, theta_1, 80)
