from functools import partial
from typing import TYPE_CHECKING, Collection

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from scipy.integrate import quad

from hamilflow.models.kepler_problem.dynamics import (
    tau_of_u_elliptic,
    tau_of_u_hyperbolic,
    tau_of_u_parabolic,
    tau_of_u_prime,
)

if TYPE_CHECKING:
    from typing import Callable

    from numpy import typing as npt

_EPS_SQRT = 1e-16
_EPS_ECC = 1e-5


@pytest.fixture(params=[1 / 3, 1 / 2, 5 / 7, 1.0, 12 / 11, 29 / 13])
def ecc(request: pytest.FixtureRequest) -> float:
    return request.param


@pytest.fixture(params=[-0.7, False, 0.7])
def u_s(request: pytest.FixtureRequest, ecc: float) -> "npt.ArrayLike":
    # There are dividends sqrt(e**2 - u**2) and (u + 1),
    # hence u cannot be too close to +e / -e / -1
    f = 1 - _EPS_ECC
    ecc_f = ecc * f
    return max(-f, request.param * ecc) or np.linspace(max(-f, -ecc_f) * f, ecc_f, 7)


class TestTauOfU:
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

    def test_const(
        self, ecc: float, tau_of_u: "Callable[[float, npt.ArrayLike], npt.ArrayLike]"
    ) -> None:
        # There are dividends sqrt(e**2 - u**2) and (u + 1),
        # hence u cannot be too close to +e / -e / -1
        res = np.array(tau_of_u(ecc, ecc * (1 - _EPS_SQRT)), copy=False)
        almost_zero = np.full(res.shape, _EPS_SQRT)
        assert_array_almost_equal(res, almost_zero)

    def test_tau_of_u(
        self,
        ecc: float,
        u_s: "npt.ArrayLike",
        tau_of_u: "Callable[[float, npt.ArrayLike], npt.ArrayLike]",
    ) -> None:
        def integrand(u: "npt.ArrayLike") -> "npt.ArrayLike":
            return tau_of_u_prime(ecc, u)

        u_s = np.array(u_s, copy=False).reshape(-1)
        rets = [quad(integrand, ecc, u) for u in u_s]
        integrals = np.array([ret[0] for ret in rets])
        assert_allclose(integrals, np.array(tau_of_u(ecc, u_s), copy=False))
