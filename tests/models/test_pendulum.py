import math
from typing import Sequence

import numpy as np
import pytest
from numpy import typing as npt
from numpy.testing import assert_array_almost_equal

from hamilflow.models.pendulum import Pendulum, PendulumIC, PendulumSystem


@pytest.fixture(params=[0.3, 0.6, 1.5])
def omega0(request: pytest.FixtureRequest) -> float:
    return request.param


@pytest.fixture(params=[-1.5, -0.3, 0.2, 0.4, +1.4])
def theta0(request: pytest.FixtureRequest) -> float:
    return request.param


@pytest.fixture(params=(-1.2, [-5.5, 0.0, 0.5, 1.0], np.linspace(0, 1, 7)))
def times(request: pytest.FixtureRequest) -> float:
    return request.param


class TestPendulumSystem:
    @pytest.mark.parametrize("omega0", [-1.0, 0])
    def test_omega0_range(self, omega0: float) -> None:
        with pytest.raises(ValueError):
            _ = PendulumSystem(omega0=omega0)


class TestPendulumIC:
    @pytest.mark.parametrize("theta0", [-2.0, 2.0])
    def test_theta0_range(self, theta0: float) -> None:
        with pytest.raises(ValueError):
            _ = PendulumIC(theta0=theta0)

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

    def test_transf(
        self,
        omega0: float,
        theta0: float,
        times: "Sequence[float] | npt.ArrayLike",
    ) -> None:
        p = Pendulum(omega0, theta0)
        arr_times = np.asarray(times)

        sin_u = np.sin(p.u(arr_times))
        theta_terms = np.sin(p.theta(arr_times) / 2) / p._k
        assert_array_almost_equal(theta_terms, sin_u)

    def test_period_dynamic_theta(
        self,
        omega0: float,
        theta0: float,
        times: "Sequence[float] | npt.ArrayLike",
    ) -> None:
        p = Pendulum(omega0, theta0)
        arr_times_1 = np.array(times, copy=False) + p.period

        theta, theta_1 = p.theta(times), p.theta(arr_times_1)

        assert_array_almost_equal(theta, theta_1)

    def test_generate_from(self, omega0: float, theta0: float) -> None:
        p = Pendulum(omega0, theta0)

        df = p.generate_from(n_periods=1, n_samples_per_period=10)
        assert len(df) == 10
