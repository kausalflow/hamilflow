"""Exact solution of Kepler dynamics."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Callable

    from numpy import typing as npt

_1_3 = 1 / 3


def tau_of_u_exact_elliptic(
    ecc: float,
    u: "npt.NDArray[np.float64]",
) -> "npt.NDArray[np.float64]":
    r"""Exact solution for tau of u in the elliptic case.

    For $-e \le u \le e$,
    $$ \tau = -\frac{\sqrt{e^2-u^2}}{(1-e^2)(1+u)}
    + \frac{\frac{\pi}{2} - \arctan\frac{e^2+u}{\sqrt{(1-e^2)(e^2-u^2)}}}{(1-e^2)^{\frac{3}{2}}}\,. $$
    """
    cosqr, eusqrt = 1 - ecc**2, np.sqrt(ecc**2 - u**2)
    trig_numer = np.pi / 2 - np.arctan((ecc**2 + u) / np.sqrt(cosqr) / eusqrt)
    return -eusqrt / cosqr / (1 + u) + trig_numer / cosqr**1.5


def tau_of_e_plus_u_elliptic(
    ecc: float,
    u: "npt.NDArray[np.float64]",
) -> "npt.NDArray[np.float64]":
    r"""Expansion for tau of u in the ellpitic case at $u = -e+0$.

    The exact solution has a removable singularity at $u = -e$, hence this
    expansion helps with numerics.

    Let $\epsilon = \sqrt{\frac{2(e+u)}{e}}$,
    $$ \tau = \frac{\pi}{(1-e^2)^\frac{3}{2}}
    - \frac{1}{(1-e)^2}\epsilon
    - \frac{1-9e}{24(1-e)^3}\epsilon^3
    + O\left(\epsilon^5\right)\,. $$
    """
    epu = np.sqrt(2 * (ecc + u) / ecc)
    const = np.pi / (1 - ecc**2) ** 1.5
    return const - epu / (ecc - 1) ** 2 - epu**3 * (1 - 9 * ecc) / 24 / (1 - ecc) ** 3


def tau_of_e_minus_u_elliptic(
    ecc: float,
    u: "npt.NDArray[np.float64]",
) -> "npt.NDArray[np.float64]":
    r"""Expansion for tau of u in the ellpitic case at $u = +e-0$.

    The exact solution has a removable singularity at $u = +e$, hence this
    expansion helps with numerics.

    Let $\epsilon = \sqrt{\frac{2(e-u)}{e}}$,
    $$ \tau = \frac{1}{(1+e)^2}\epsilon
    - \frac{1+9e}{24(1+e)^3}\epsilon^3
    + O\left(\epsilon^5\right)\,. $$
    """
    emu = np.sqrt(2 * (ecc - u) / ecc)
    return emu / (1 + ecc) ** 2 - emu**3 * (1 + 9 * ecc) / 24 / (1 + ecc) ** 3


def tau_of_u_elliptic(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
    """Calculate the scaled time tau from u in the elliptic case.

    :param ecc: eccentricity, 0 < ecc < 1 (unchecked)
    :param u: convenient radial inverse
    :return: scaled time tau
    """
    return _approximate_at_termina(
        ecc,
        u,
        tau_of_u_exact_elliptic,
        tau_of_e_plus_u_elliptic,
        tau_of_e_minus_u_elliptic,
    )


def tau_of_u_parabolic(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
    r"""Calculate the scaled time tau from u in the parabolic case.

    For $-1 < u \le 1$,
    $$ \tau = \frac{\sqrt{1-u}(2+u)}{3(1+u)^\frac{3}{2}}\,. $$

    :param ecc: eccentricity, ecc == 1 (unchecked, unused)
    :param u: convenient radial inverse
    :return: scaled time tau
    """
    u = np.asarray(u)
    return np.sqrt(1 - u) * (2 + u) / 3 / (1 + u) ** 1.5


def tau_of_u_exact_hyperbolic(
    ecc: float,
    u: "npt.ArrayLike",
) -> "npt.NDArray[np.float64]":
    r"""Exact solution for tau of u in the hyperbolic case.

    For $-1 < u \le e$,
    $$ \tau = \frac{\sqrt{e^2-u^2}}{(e^2-1)(1+u)}
    - \frac{\mathrm{artanh}\frac{e^2+u}{\sqrt{(e^2-1)(e^2-u^2)}}}{(e^2-1)^{\frac{3}{2}}}\,. $$
    """
    u = np.asarray(u)
    cosqr, eusqrt = ecc**2 - 1, np.sqrt(ecc**2 - u**2)
    trig_numer = np.arctanh(np.sqrt(cosqr) * eusqrt / (ecc**2 + u))
    return eusqrt / cosqr / (1 + u) - trig_numer / cosqr**1.5


def _tau_of_1_plus_u_hyperbolic(
    ecc: float,
    u: "npt.NDArray[np.float64]",
) -> "npt.NDArray[np.float64]":
    cosqr = ecc**2 - 1
    up1 = ecc * (1 + u) / 2 / cosqr
    diverge = np.log(up1) + ecc / 2 / up1
    regular = 1 - (2 + ecc**2) / ecc * up1 + (3 + 2 / ecc**2) * up1**2
    return (diverge + regular) / cosqr**1.5


def _tau_of_e_minus_u_hyperbolic(
    ecc: float,
    u: "npt.NDArray[np.float64]",
) -> "npt.NDArray[np.float64]":
    emu = np.sqrt(2 * (ecc - u) / ecc)
    return emu / (1 + ecc) ** 2 + emu**3 * (1 + 9 * ecc) / 24 / (1 + ecc) ** 3


def tau_of_u_hyperbolic(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
    """Calculate the scaled time tau from u in the hyperbolic case.

    :param ecc: eccentricity, ecc > 1 (unchecked)
    :param u: convenient radial inverse
    :return: scaled time tau
    """
    return _approximate_at_termina(
        ecc,
        u,
        tau_of_u_exact_hyperbolic,
        _tau_of_1_plus_u_hyperbolic,
        _tau_of_e_minus_u_hyperbolic,
    )


def tau_of_u_prime(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
    """Calculate the first derivative of scaled time tau with respect to u.

    :param ecc: eccentricity, ecc >= 0 (unchecked)
    :param u: convenient radial inverse
    :return: the first derivative scaled time tau with respect to u
    """
    u = np.asarray(u)
    return -1 / (1 + u) ** 2 / np.sqrt(ecc**2 - u**2)


def tau_of_u_prime2(ecc: float, u: "npt.ArrayLike") -> "npt.NDArray[np.float64]":
    """Calculate the second derivative of scaled time tau with respect to u.

    :param ecc: eccentricity, ecc >= 0 (unchecked)
    :param u: convenient radial inverse
    :return: the second derivative scaled time tau with respect to u
    """
    u = np.asarray(u)
    u2 = u**2
    return (2 * ecc**2 - u - 3 * u2) / (1 + u) ** 3 / (ecc**2 - u2) ** 1.5


def esolve_u_from_tau_parabolic(
    ecc: float,
    tau: "npt.ArrayLike",
) -> "npt.NDArray[np.float64]":
    r"""Calculate the convenient radial inverse u from tau in the parabolic case, using the exact solution.

    Let $T = 1+9\tau^2$,
    $$ u = -1
    + \left(\frac{3\tau}{T^\frac{3}{2}} + \frac{1}{T}\right)^\frac{1}{3}
    + \left(\frac{1}{3\tau T^\frac{3}{2}+T^2}\right)^\frac{1}{3}\,. $$

    :param ecc: eccentricity, ecc = 0 (unchecked, unused)
    :param tau: scaled time
    :return: convenient radial inverse u
    """
    tau = np.asarray(tau)
    tau_3 = 3 * tau
    term = 1 + tau_3**2  # 1 + 9 * tau**2
    term1_5 = term**1.5  # (1 + 9 * tau**2)**1.5
    second_term = (tau_3 / term1_5 + 1 / term) ** _1_3
    third_term = 1 / (tau_3 * term1_5 + term**2) ** _1_3
    return -1 + second_term + third_term


def _approximate_at_termina(
    ecc: float,
    u: "npt.ArrayLike",
    exact: "Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]",
    left: "Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]",
    right: "Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]",
):
    u = np.asarray(u)
    u_s = u.reshape(-1)
    res = exact(ecc, u_s)
    mask = np.isnan(res)
    u_masked = u_s[mask]
    res[mask] = np.where(u_masked < 0, left(ecc, u_masked), right(ecc, u_masked))
    return res.reshape(u.shape)
