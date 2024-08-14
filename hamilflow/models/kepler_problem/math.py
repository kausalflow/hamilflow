import math
from typing import Callable

import numpy as np
from numpy import typing as npt
from scipy.optimize._root_scalar import root_scalar


def tau_of_u_root_elliptic(e: float, tau: float, u: float) -> float:
    cosqr, eusqrt = 1 - e**2, math.sqrt(e**2 - u**2)
    return (
        -eusqrt / cosqr / (1 + u)
        - math.atan((e**2 + u) / math.sqrt(cosqr) / eusqrt) / cosqr**1.5
        - tau
    )


def tau_of_u_root_parabolic(e: float, tau: float, u: float) -> float:
    return math.sqrt(1 - u) * (2 + u) / 3 / (1 + u) ** 1.5 - tau


def tau_of_u_root_hyperbolic(e: float, tau: float, u: float) -> float:
    cosqr, eusqrt = e**2 - 1, math.sqrt(e**2 - u**2)
    return (
        eusqrt / cosqr / (1 + u)
        - math.atanh(math.sqrt(cosqr) * eusqrt / (e**2 + u)) / cosqr**1.5
        - tau
    )


def tau_of_u_prime(e: float, u: float) -> float:
    return 1 / (1 + u) ** 2 / math.sqrt(e**2 - u**2)


def u_of_tau_by_inverse(
    tau_of_u_root: Callable[[float, float, float], float], e: float, tau: float
) -> float:
    return root_scalar(
        tau_of_u_root, (e, tau), "newton", x0=0.0, fprime=tau_of_u_prime
    ).root


def acos_with_shift(x: npt.ArrayLike, shift: npt.ArrayLike) -> npt.ArrayLike:
    p_shift = (div := np.floor(shift)) * 2 * np.pi
    remainder = shift - div
    p_value = np.arccos(x)
    return p_shift + np.where(remainder <= 0.5, p_value, 2 * np.pi - p_value)
