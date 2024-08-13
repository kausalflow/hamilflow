import math
from typing import Callable

from numba import njit
from scipy.optimize._root_scalar import root_scalar


@njit
def tau_of_u_root_elliptic(e: float, tau: float, u: float) -> float:
    cosqr, eusqrt = 1 - e**2, math.sqrt(e**2 - u**2)
    return (
        -eusqrt / cosqr / (1 + u)
        - math.atan((e**2 + u) / math.sqrt(cosqr) / eusqrt) / cosqr**1.5
        - tau
    )


@njit
def tau_of_u_root_parabolic(tau: float, u: float) -> float:
    return math.sqrt(1 - u) * (2 + u) / 3 / (1 + u) ** 1.5 - tau


@njit
def tau_of_u_root_hyperbolic(e: float, tau: float, u: float) -> float:
    cosqr, eusqrt = e**2 - 1, math.sqrt(e**2 - u**2)
    return (
        eusqrt / cosqr / (1 + u)
        - math.atanh(math.sqrt(cosqr) * eusqrt / (e**2 + u)) / cosqr**1.5
        - tau
    )


@njit
def tau_of_u_prime(e: float, u: float) -> float:
    return 1 / (1 + u) ** 2 / math.sqrt(e**2 - u**2)


@njit
def u_of_tau_by_inverse(
    tau_of_u_root: Callable[[float, float], float], e: float, tau: float
) -> float:
    return root_scalar(
        tau_of_u_root, (e, tau), "newton", x0=0.0, fprime=tau_of_u_prime
    ).root
