from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy import typing as npt

_1_3 = 1 / 3


def _tau_of_u_exact_elliptic(
    ecc: float, u: "npt.NDArray[np.float64]"
) -> "npt.NDArray[np.float64]":
    cosqr, eusqrt = 1 - ecc**2, np.sqrt(ecc**2 - u**2)
    trig_numer = np.pi / 2 - np.arctan((ecc**2 + u) / np.sqrt(cosqr) / eusqrt)
    return -eusqrt / cosqr / (1 + u) + trig_numer / cosqr**1.5


def _tau_of_e_plus_u_elliptic(
    ecc: float, u: "npt.NDArray[np.float64]"
) -> "npt.NDArray[np.float64]":
    epu = np.sqrt(2 * (ecc + u) / ecc)
    const = np.pi / (1 - ecc**2) ** 1.5
    return const - epu / (ecc - 1) ** 2 + epu**3 * (1 - 9 * ecc) / 24 / (1 - ecc) ** 3


def _tau_of_e_minus_u_elliptic(
    ecc: float, u: "npt.NDArray[np.float64]"
) -> "npt.NDArray[np.float64]":
    emu = np.sqrt(2 * (ecc - u) / ecc)
    return emu / (1 + ecc) ** 2 - emu**3 * (1 + 9 * ecc) / 24 / (1 + ecc) ** 3


def tau_of_u_elliptic(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
    u = np.array(u, copy=False)
    u_s = u.reshape(-1)
    res = _tau_of_u_exact_elliptic(ecc, u_s)
    mask = np.isnan(res)
    res[mask] = np.where(
        u_s[mask] < 0,
        _tau_of_e_plus_u_elliptic(ecc, u_s[mask]),
        _tau_of_e_minus_u_elliptic(ecc, u_s[mask]),
    )
    return res.reshape(u.shape)


def tau_of_u_parabolic(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
    u = np.array(u, copy=False)
    return np.sqrt(1 - u) * (2 + u) / 3 / (1 + u) ** 1.5


def tau_of_u_hyperbolic(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
    u = np.array(u, copy=False)
    cosqr, eusqrt = ecc**2 - 1, np.sqrt(ecc**2 - u**2)
    trig_numer = np.arctanh(np.sqrt(cosqr) * eusqrt / (ecc**2 + u))
    return eusqrt / cosqr / (1 + u) - trig_numer / cosqr**1.5


def tau_of_u_prime(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
    u = np.array(u, copy=False)
    return -1 / (1 + u) ** 2 / np.sqrt(ecc**2 - u**2)


def tau_of_u_prime2(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
    u = np.array(u, copy=False)
    u2 = u**2
    return (2 * ecc**2 - u - 3 * u2) / (1 + u) ** 3 / (ecc**2 - u2) ** 1.5


def esolve_u_from_tau_parabolic(
    ecc: float, tau: "npt.ArrayLike"
) -> "npt.NDArray[np.float64]":
    tau = np.array(tau, copy=False)
    tau_3 = 3 * tau
    term = 1 + tau_3**2  # 1 + 9 * tau**2
    term1_5 = term**1.5  # (1 + 9 * tau**2)**1.5
    second_term = (tau_3 / term1_5 + 1 / term) ** _1_3
    third_term = 1 / (tau_3 * term1_5 + term**2) ** _1_3
    return -1 + second_term + third_term
