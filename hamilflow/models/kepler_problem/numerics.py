"""Numerics for the Kepler problem."""

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import OptimizeResult, newton, toms748

from hamilflow.models.kepler_problem.dynamics import (
    esolve_u_from_tau_parabolic,
    tau_of_u_elliptic,
    tau_of_u_hyperbolic,
    tau_of_u_prime,
    tau_of_u_prime2,
)

if TYPE_CHECKING:
    from numpy import typing as npt


def _u0_elliptic(
    ecc: float,
    tau: "npt.NDArray[np.float64]",
) -> "npt.NDArray[np.float64]":
    cosqr = 1 - ecc**2
    cot = 1 / np.tan(cosqr**1.5 * tau)
    return ecc * (-ecc + cosqr * cot * np.sqrt(1 + cot**2)) / (1 + cosqr * cot**2)


def _u0_hyperbolic(
    ecc: float,
    tau: "npt.NDArray[np.float64]",
) -> "npt.NDArray[np.float64]":
    cosqr, tau2 = ecc**2 - 1, tau**2
    numer = -(cosqr**2) * tau2 + np.sqrt(ecc**2 + cosqr**3 * tau2)
    return numer / (1 + cosqr**2 * tau2)


def nsolve_u_from_tau_newton(ecc: float, tau: "npt.ArrayLike") -> OptimizeResult:
    tau = np.array(tau, copy=False)
    if 0 < ecc < 1:
        tau_of_u = tau_of_u_elliptic
        u0 = _u0_elliptic(ecc, tau)
    elif ecc > 1:
        tau_of_u = tau_of_u_hyperbolic
        u0 = _u0_hyperbolic(ecc, tau)
    else:
        raise ValueError(f"Expect ecc > 0, ecc != 1, got {ecc}")

    def f(u: float, tau: "npt.NDArray[np.float64]") -> "npt.NDArray[np.float64]":
        return tau_of_u(ecc, u) - tau

    def fprime(u: float, tau: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
        return tau_of_u_prime(ecc, u)

    def fprime2(u: float, tau: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
        return tau_of_u_prime2(ecc, u)

    return newton(f, u0, fprime, (tau,), fprime2=fprime2, full_output=True, disp=False)


def nsolve_u_from_tau_bisect(ecc: float, tau: "npt.ArrayLike") -> list[OptimizeResult]:
    tau_s = np.array(tau, copy=False).reshape(-1)
    if 0 < ecc < 1:
        tau_of_u = tau_of_u_elliptic
    elif ecc > 1:
        tau_of_u = tau_of_u_hyperbolic
    else:
        raise ValueError(f"Expect ecc > 0, ecc != 1, got {ecc}")

    def f(u: float, tau: float) -> np.float64:
        return (
            np.finfo(np.float64).max if u == -1 else np.float64(tau_of_u(ecc, u) - tau)
        )

    return [toms748(f, max(-1, -ecc), ecc, (ta,), 2, full_output=True) for ta in tau_s]


def u_of_tau(
    ecc: float,
    tau: "npt.ArrayLike",
    method: Literal["bisect", "newton"] = "bisect",
) -> "npt.NDArray[np.float64]":
    tau = np.array(tau, copy=False)
    if ecc == 0:
        return np.zeros(tau.shape)
    elif ecc == 1:
        return esolve_u_from_tau_parabolic(ecc, tau)
    elif ecc > 0:
        match method:
            case "bisect":
                return np.array([s[0] for s in nsolve_u_from_tau_bisect(ecc, tau)])
            case "newton":
                return nsolve_u_from_tau_newton(ecc, tau)[0]
            case _:
                raise ValueError(f"Expect 'bisect' or 'newton', got {method}")
    else:
        raise ValueError(f"Expect ecc >= 0, got {ecc}")
