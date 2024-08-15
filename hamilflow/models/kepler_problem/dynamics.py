from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.optimize._root_scalar import root_scalar

if TYPE_CHECKING:
    from numpy import typing as npt


def tau_of_u_elliptic(ecc: float, u: "npt.ArrayLike") -> "npt.ArrayLike":
    cosqr, eusqrt = 1 - ecc**2, np.sqrt(ecc**2 - u**2)
    return (
        -eusqrt / cosqr / (1 + u)
        - np.arctan((ecc**2 + u) / np.sqrt(cosqr) / eusqrt) / cosqr**1.5
    )


def tau_of_u_parabolic(ecc: float, u: "npt.ArrayLike") -> "npt.ArrayLike":
    return np.sqrt(1 - u) * (2 + u) / 3 / (1 + u) ** 1.5


def tau_of_u_hyperbolic(ecc: float, u: "npt.ArrayLike") -> "npt.ArrayLike":
    cosqr, eusqrt = ecc**2 - 1, np.sqrt(ecc**2 - u**2)
    return (
        eusqrt / cosqr / (1 + u)
        - np.arctanh(np.sqrt(cosqr) * eusqrt / (ecc**2 + u)) / cosqr**1.5
    )


def tau_of_u_prime(ecc: float, u: "npt.ArrayLike") -> "npt.ArrayLike":
    return -1 / (1 + u) ** 2 / np.sqrt(ecc**2 - u**2)


def solve_u_of_tau(
    tau_of_u_eq: "Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]",
    tau: "npt.NDArray[np.float32]",
) -> "npt.NDArray[np.float32]":
    return np.array(
        [
            root_scalar(tau_of_u_eq, (ta,), "newton", x0=0.0, fprime=tau_of_u_prime)
            for ta in tau
        ]
    )


def acos_with_shift(x: "npt.ArrayLike", shift: "npt.ArrayLike") -> "npt.ArrayLike":
    p_shift = (div := np.floor(shift)) * 2 * np.pi
    remainder = shift - div
    p_value = np.arccos(x)
    return p_shift + np.where(remainder <= 0.5, p_value, 2 * np.pi - p_value)
