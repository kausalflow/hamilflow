from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import OptimizeResult, newton

if TYPE_CHECKING:
    from numpy import typing as npt

_1_3 = 1 / 3


def tau_of_u_elliptic(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float32]":
    u = np.array(u, copy=False)
    cosqr, eusqrt = 1 - ecc**2, np.sqrt(ecc**2 - u**2)
    return (
        -eusqrt / cosqr / (1 + u)
        + (np.pi / 2 - np.arctan((ecc**2 + u) / np.sqrt(cosqr) / eusqrt)) / cosqr**1.5
    )


def tau_of_u_parabolic(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float32]":
    u = np.array(u, copy=False)
    return np.sqrt(1 - u) * (2 + u) / 3 / (1 + u) ** 1.5


def tau_of_u_hyperbolic(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float32]":
    u = np.array(u, copy=False)
    cosqr, eusqrt = ecc**2 - 1, np.sqrt(ecc**2 - u**2)
    return (
        eusqrt / cosqr / (1 + u)
        - np.arctanh(np.sqrt(cosqr) * eusqrt / (ecc**2 + u)) / cosqr**1.5
    )


def tau_of_u_prime(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float32]":
    u = np.array(u, copy=False)
    return -1 / (1 + u) ** 2 / np.sqrt(ecc**2 - u**2)


def tau_of_u_prime2(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float32]":
    u = np.array(u, copy=False)
    u2 = u**2
    return (2 * ecc**2 - u - 3 * u2) / (1 + u) ** 3 / (ecc**2 - u2) ** 1.5


def _u0_elliptic(
    ecc: float, tau: "npt.NDArray[np.float32]"
) -> "npt.NDArray[np.float32]":
    cosqr = 1 - ecc**2
    cot = 1 / np.tan(cosqr**1.5 * tau)
    return ecc * (-ecc + cosqr * cot * np.sqrt(1 + cot**2)) / (1 + cosqr * cot**2)


def esolve_u_from_tau_parabolic(
    ecc: float, tau: "npt.ArrayLike"
) -> "npt.NDArray[np.float32]":
    tau = np.array(tau, copy=False)
    tau_3 = 3 * tau
    term = 1 + tau_3**2  # 1 + 9 * tau**2
    term1_5 = term**1.5  # (1 + 9 * tau**2)**1.5

    return (
        -1
        + (tau_3 / term1_5 + 1 / term) ** _1_3
        + 1 / (tau_3 * term1_5 + term**2) ** _1_3
    )


def _u0_hyperbolic(
    ecc: float, tau: "npt.NDArray[np.float32]"
) -> "npt.NDArray[np.float32]":
    cosqr = ecc**2 - 1
    tau2 = tau**2
    return (-(cosqr**2) * tau2 + np.sqrt(ecc**2 + cosqr**3 * tau2)) / (
        1 + cosqr**2 * tau2
    )


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

    def f(u: float, tau: "npt.ArrayLike") -> "npt.NDArray[np.float32]":
        return tau_of_u(ecc, u) - np.array(tau, copy=False)

    def fprime(u: float, tau: "npt.ArrayLike") -> "npt.NDArray[np.float32]":
        return tau_of_u_prime(ecc, u)

    def fprime2(u: float, tau: "npt.ArrayLike") -> "npt.NDArray[np.float32]":
        return tau_of_u_prime2(ecc, u)

    return newton(f, u0, fprime, (tau,), fprime2=fprime2, full_output=True, disp=False)


def u_of_tau(ecc: float, tau: "npt.ArrayLike") -> "npt.NDArray[np.float32]":
    if ecc < 1:
        return nsolve_u_from_tau(ecc, tau)[0]
    elif ecc == 1:
        return esolve_u_from_tau_parabolic(ecc, tau)
    elif ecc > 1:
        return nsolve_u_from_tau(ecc, tau)[0]
    else:
        raise ValueError
