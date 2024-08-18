from typing import TYPE_CHECKING, Literal

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


# 5 / 7, 12 / 11, 257 / 13 makes u_of_tau from newton fail
@pytest.fixture(params=[0.0, 0.1, 0.3, 0.7, 0.9, 1.0, 1.1, 2.0, 11.0, 101.0])
def ecc(request: pytest.FixtureRequest) -> float:
    return request.param


@pytest.fixture(params=[-0.9, False, 0.9])
def u_s(request: pytest.FixtureRequest, ecc: float) -> "npt.ArrayLike":
    # There are dividends sqrt(e**2 - u**2) and (u + 1),
    # hence u cannot be too close to +e / -e / -1
    f = 1 - _EPS_ECC
    ecc_f = ecc * f
    return max(-f, request.param * ecc_f) or np.linspace(max(-f, -ecc_f), ecc_f, 7)


@pytest.fixture()
def tau_of_u(ecc: float) -> "Callable[[float, npt.ArrayLike], npt.ArrayLike]":
    if ecc == 0:
        pytest.skip(f"Circular case")
    elif 0 < ecc < 1:
        return tau_of_u_elliptic
    elif ecc == 1:
        return tau_of_u_parabolic
    elif ecc > 1:
        return tau_of_u_hyperbolic
    else:
        raise ValueError(f"Expect ecc >= 0, got {ecc}")


class TestUOfTau:
    @pytest.mark.parametrize(
        "method",
        [
            "bisect",
            pytest.param(
                "newton",
                marks=pytest.mark.xfail(
                    reason="Newton method gives nan's, possibly because of inefficient initial estimate"
                ),
            ),
        ],
    )
    def test_u_of_tau(
        self,
        ecc: float,
        method: Literal["bisect", "newton"],
        u_s: "npt.ArrayLike",
        tau_of_u: "Callable[[float, npt.ArrayLike], npt.ArrayLike]",
    ) -> None:
        u_s = np.array(u_s, copy=False)
        tau = tau_of_u(ecc, u_s)
        actual = u_of_tau(ecc, tau, method)
        assert_array_almost_equal(u_s, actual)
