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
    """Calculate the convenient radial inverse u from tau in the elliptic or parabolic case, using the Newton method.

    :param ecc: eccentricity, ecc > 0, ecc != 1
    :param tau: scaled time
    :raises ValueError: when `ecc` is invalid
    :return: numeric OptimizeResult from scipy
    """
    tau = np.asarray(tau)
    if 0 < ecc < 1:
        tau_of_u = tau_of_u_elliptic
        u0 = _u0_elliptic(ecc, tau)
    elif ecc > 1:
        tau_of_u = tau_of_u_hyperbolic
        u0 = _u0_hyperbolic(ecc, tau)
    else:
        msg = f"Expect ecc > 0, ecc != 1, got {ecc}"
        raise ValueError(msg)

    def f(u: float, tau: "npt.NDArray[np.float64]") -> "npt.NDArray[np.float64]":
        return tau_of_u(ecc, u) - tau

    def fprime(
        u: float,
        tau: "npt.ArrayLike",  # noqa: ARG001
    ) -> "npt.NDArray[np.float64]":
        return tau_of_u_prime(ecc, u)

    def fprime2(
        u: float,
        tau: "npt.ArrayLike",  # noqa: ARG001
    ) -> "npt.NDArray[np.float64]":
        return tau_of_u_prime2(ecc, u)

    return newton(f, u0, fprime, (tau,), fprime2=fprime2, full_output=True, disp=False)


def nsolve_u_from_tau_bisect(ecc: float, tau: "npt.ArrayLike") -> list[OptimizeResult]:
    """Calculate the convenient radial inverse u from tau in the elliptic or parabolic case, using the bisect method.

    :param ecc: eccentricity, ecc > 0, ecc != 1
    :param tau: scaled time
    :return: numeric OptimizeResult from scipy
    """
    tau_s = np.asarray(tau).reshape(-1)
    if 0 < ecc < 1:
        tau_of_u = tau_of_u_elliptic
    elif ecc > 1:
        tau_of_u = tau_of_u_hyperbolic
    else:
        msg = f"Expect ecc > 0, ecc != 1, got {ecc}"
        raise ValueError(msg)

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
    """Calculate the convenient radial inverse u from tau, using numeric methods.

    :param ecc: eccentricity, ecc >= 0
    :param tau: scaled time
    :param method: "newton" or "bisect" numeric methods, defaults to "bisect"
    :raises ValueError: when `method` is invalid
    :raises ValueError: when `ecc` is invalid
    :return: convenient radial inverse u
    """
    tau = np.asarray(tau)
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
                msg = f"Expect 'bisect' or 'newton', got {method}"
                raise ValueError(msg)
    else:
        msg = f"Expect ecc >= 0, got {ecc}"
        raise ValueError(msg)
