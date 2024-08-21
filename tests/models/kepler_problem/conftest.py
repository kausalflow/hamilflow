from typing import TYPE_CHECKING

import numpy as np
import pytest

from hamilflow.models.kepler_problem.dynamics import (
    tau_of_u_elliptic,
    tau_of_u_hyperbolic,
    tau_of_u_parabolic,
)

if TYPE_CHECKING:
    from typing import Callable

    from numpy import typing as npt


_EPS_ECC = 1e-5


@pytest.fixture(
    scope="module",
    params=[0.0, 0.1, 0.3, 0.7, 0.9, 1.0, 1.1, 2.0, 11.0, 101.0],
)
def ecc(request: pytest.FixtureRequest) -> float:
    return request.param


@pytest.fixture(scope="module", params=[-0.9, False, 0.9])
def u_s(request: pytest.FixtureRequest, ecc: float) -> "npt.ArrayLike":
    # There are dividends sqrt(e**2 - u**2) and (u + 1),
    # hence u cannot be too close to +e / -e / -1
    f = 1 - _EPS_ECC
    ecc_f = ecc * f
    return max(-f, request.param * ecc_f) or np.linspace(max(-f, -ecc_f), ecc_f, 7)


@pytest.fixture(scope="module")
def tau_of_u(ecc: float) -> "Callable[[float, npt.ArrayLike], npt.ArrayLike]":
    if ecc == 0:
        pytest.skip("Circular case")
    elif 0 < ecc < 1:
        return tau_of_u_elliptic
    elif ecc == 1:
        return tau_of_u_parabolic
    elif ecc > 1:
        return tau_of_u_hyperbolic
    else:
        raise ValueError(f"Expect ecc >= 0, got {ecc}")
