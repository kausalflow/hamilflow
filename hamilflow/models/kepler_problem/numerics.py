from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import OptimizeResult, newton, toms748

from hamilflow.models.kepler_problem.dynamics import (
    tau_of_u_elliptic,
    tau_of_u_hyperbolic,
    tau_of_u_parabolic,
    tau_of_u_prime,
    tau_of_u_prime2,
    esolve_u_from_tau_parabolic)

if TYPE_CHECKING:
    from numpy import typing as npt

def _u0_elliptic(
    ecc: float, tau: "npt.NDArray[np.float64]"
) -> "npt.NDArray[np.float64]":
    cosqr = 1 - ecc**2
    cot = 1 / np.tan(cosqr**1.5 * tau)
    return ecc * (-ecc + cosqr * cot * np.sqrt(1 + cot**2)) / (1 + cosqr * cot**2)

def _u0_hyperbolic(
    ecc: float, tau: "npt.NDArray[np.float64]"
) -> "npt.NDArray[np.float64]":
    cosqr, tau2 = ecc**2 - 1, tau**2
    numer = -(cosqr**2) * tau2 + np.sqrt(ecc**2 + cosqr**3 * tau2)
    return numer / (1 + cosqr**2 * tau2)


def nsolve_u_from_tau(ecc: float, tau: "npt.ArrayLike") -> OptimizeResult:
    tau = np.array(tau, copy=False)
    if ecc < 1:
        tau_of_u = tau_of_u_elliptic
        u0 = _u0_elliptic(ecc, tau)
    elif ecc > 1:
        tau_of_u = tau_of_u_hyperbolic
        u0 = _u0_hyperbolic(ecc, tau)
    else:
        raise ValueError

    def f(u: float, tau: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
        return tau_of_u(ecc, u) - np.array(tau, copy=False)

    def fprime(u: float, tau: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
        return tau_of_u_prime(ecc, u)

    def fprime2(u: float, tau: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
        return tau_of_u_prime2(ecc, u)

    print("start", u0)
    return newton(f, u0, fprime, (tau,), fprime2=fprime2, maxiter=1000, full_output=True, disp=False)
    # return toms748(f, max(-1, -ecc), ecc, (tau,), 2, full_output=True)


def u_of_tau(ecc: float, tau: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
    if ecc == 1:
        return esolve_u_from_tau_parabolic(ecc, tau)
    elif ecc > 0:
        return nsolve_u_from_tau(ecc, tau)[0]
    else:
        raise ValueError
