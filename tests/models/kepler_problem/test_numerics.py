from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from hamilflow.models.kepler_problem.dynamics import (
    tau_of_u_elliptic,
    tau_of_u_hyperbolic,
    tau_of_u_parabolic,
)
from hamilflow.models.kepler_problem.numerics import u_of_tau

if TYPE_CHECKING:
    from typing import Callable

    from numpy import typing as npt


_EPS_ECC = 1e-5


# 5 / 7, 12 / 11, 257 / 13 makes test_u_of_tau fail
@pytest.fixture(params=[1 / 3, 1 / 2, 1.0, 29 / 13, 256 / 19])
def ecc(request: pytest.FixtureRequest) -> float:
    return request.param


@pytest.fixture(params=[-0.9, False, 0.9])
def u_s(request: pytest.FixtureRequest, ecc: float) -> "npt.ArrayLike":
    # There are dividends sqrt(e**2 - u**2) and (u + 1),
    # hence u cannot be too close to +e / -e / -1
    f = 1 - _EPS_ECC
    ecc_f = ecc * f
    return max(-f, request.param * ecc) or np.linspace(max(-f, -ecc_f), ecc_f, 7)


class TestUOfTau:
    @pytest.fixture()
    def tau_of_u(self, ecc: float) -> "Callable[[float, npt.ArrayLike], npt.ArrayLike]":
        if 0 < ecc < 1:
            return tau_of_u_elliptic
        elif ecc == 1:
            return tau_of_u_parabolic
        elif ecc > 1:
            return tau_of_u_hyperbolic
        else:
            raise ValueError(f"Expected ecc > 0, got {ecc}")

    def test_u_of_tau(
        self,
        ecc: float,
        tau_of_u: "Callable[[float, npt.ArrayLike], npt.ArrayLike]",
        u_s: "npt.ArrayLike",
    ) -> None:
        u_s = np.array(u_s, copy=False)
        tau = tau_of_u(ecc, u_s)
        actual = u_of_tau(ecc, tau)
        assert_array_almost_equal(u_s, actual)
