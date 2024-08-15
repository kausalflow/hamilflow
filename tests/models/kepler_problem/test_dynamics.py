from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.integrate import quad

from hamilflow.models.kepler_problem.dynamics import (
    tau_of_u_elliptic,
    tau_of_u_hyperbolic,
    tau_of_u_parabolic,
    tau_of_u_prime,
)

if TYPE_CHECKING:
    from numpy import typing as npt

_EPS = 0.05


@pytest.mark.parametrize("ecc", [1 / 3, 1 / 2, 5 / 7, 1.0, 12 / 11, 27 / 13])
def test_tau_of_u(ecc: float) -> None:
    def integrand(u: float) -> float:
        return tau_of_u_prime(ecc, u)

    if 0 < ecc < 1:
        tau_of_u = tau_of_u_elliptic
        cosqr = 1 - ecc**2
        const = -(ecc + np.arcsin(ecc) / np.sqrt(cosqr)) / cosqr
    elif ecc == 1:
        tau_of_u = tau_of_u_parabolic
        const = 2 / 3
    elif ecc > 1:
        tau_of_u = tau_of_u_hyperbolic
        cosqr = ecc**2 - 1
        const = (ecc - np.arccosh(ecc) / np.sqrt(cosqr)) / cosqr
    else:
        raise ValueError(f"Expected ecc > 0, got {ecc}")

    u_s = np.linspace(max(-1, -ecc) + _EPS, ecc - _EPS, 5)
    rets = [quad(integrand, 0, u) for u in u_s]
    integrals = np.array([ret[0] for ret in rets]) + const
    assert_array_almost_equal(integrals, tau_of_u(ecc, u_s))
