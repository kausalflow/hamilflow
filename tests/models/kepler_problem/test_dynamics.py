from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_approx_equal,
    assert_array_almost_equal,
)
from scipy.integrate import quad

from hamilflow.models.kepler_problem.dynamics import (
    _tau_of_1_plus_u_hyperbolic,
    _tau_of_e_minus_u_elliptic,
    _tau_of_e_minus_u_hyperbolic,
    _tau_of_e_plus_u_elliptic,
    _tau_of_u_exact_elliptic,
    _tau_of_u_exact_hyperbolic,
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


@pytest.fixture(params=[0.1, 0.3, 0.7, 0.9, 1.0, 1.1, 2.0, 11.0, 101.0])
def ecc(request: pytest.FixtureRequest) -> float:
    return request.param


@pytest.fixture(params=[-0.9, False, 0.9])
def u_s(request: pytest.FixtureRequest, ecc: float) -> "npt.ArrayLike":
    # There are dividends sqrt(e**2 - u**2) and (u + 1),
    # hence u cannot be too close to +e / -e / -1
    f = 1 - _EPS_ECC
    ecc_f = ecc * f
    return max(-f, request.param * ecc) or np.linspace(max(-f, -ecc_f), ecc_f, 7)


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
        desired = np.full(res.shape, _EPS_SQRT)
        assert_array_almost_equal(desired, res)

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

    @pytest.fixture()
    @pytest.mark.skipif("ecc == 1.0")
    def exact_and_approx_tau_s(
        self, ecc: float
    ) -> "tuple[Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]], Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]], Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]]":
        if 0 < ecc < 1:
            return (
                _tau_of_u_exact_elliptic,
                _tau_of_e_plus_u_elliptic,
                _tau_of_e_minus_u_elliptic,
            )
        elif ecc > 1:
            return (
                _tau_of_u_exact_hyperbolic,
                _tau_of_1_plus_u_hyperbolic,
                _tau_of_e_minus_u_hyperbolic,
            )
        elif ecc == 1:
            pytest.skip("Parabolic case is exact. Pass")
        else:
            raise ValueError(f"Expect ecc > 0, got {ecc}")

    @pytest.mark.parametrize("epsilon", [1e-7])
    @pytest.mark.skipif("ecc == 1.0")
    def test_expansion(
        self,
        ecc: float,
        epsilon: float,
        exact_and_approx_tau_s: "tuple[Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]], Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]], Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]]",
    ) -> None:
        factor = 1 - epsilon
        f, g_s = exact_and_approx_tau_s[0], exact_and_approx_tau_s[1:]
        for u, g in zip((max(-1, -ecc) * factor, ecc * factor), g_s):
            u_s = np.array(u, copy=False)
            desired, actual = f(ecc, u_s), g(ecc, u_s)
            if not np.isinf(desired):
                assert_approx_equal(actual, desired, err_msg=f"ecc={ecc}, u={u_s}")
