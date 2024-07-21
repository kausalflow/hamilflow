from functools import cached_property
from typing import Any, Collection, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pandas.api.types import is_scalar
from scipy.fft import ifft

from ..d0.complex_harmonic_oscillator import SimpleComplexHarmonicOscillator


class HarmonicOscillatorsChain:
    r"""Generate time series data for a coupled harmonic oscillator chain
    with periodic boundary condition.

    A one-dimensional circle of $N$ interacting harmonic oscillators can be described by the Lagrangian action
    $$S_L[x_i] = \int_{t_0}^{t_1}\mathbb{d} t \left\\{ \sum_{i=0}^{N-1} \frac{1}{2}m \dot x_i^2 - \frac{1}{2}m\omega^2\left(x_i - x_{i+1}\right)^2 \right\\}\\,,$$
    where $x_N \coloneqq x_0$.

    This system can be solved in terms of _travelling waves_, obtained by discrete Fourier transform.

    We can complexify the system
    $$S_L[x_i] = S_L[x_i, \phi_j] \equiv S_L[X^\ast_i, X_j] = \int_{t_0}^{t_1}\mathbb{d} t \left\\{ \frac{1}{2}m \dot X^\ast_i \delta_{ij} \dot X_j - \frac{1}{2}m X^\ast_i A_{ij} X_j\right\\}\\,,$$
    where $A_{ij} / \omega^2$ is equal to $(-2)$ if $i=j$, $1$ if $|i-j|=1$ or $|i-j|=N$, and $0$ otherwise;
    $X_i \coloneqq x_i \mathbb{e}^{-\phi_i}$, $X^\ast_i \coloneqq x_i \mathbb{e}^{+\phi_i}$.

    $A_{ij}$ can be diagonalised by the inverse discrete Fourier transform
    $$X_i = (F^{-1})_{ik} Y_k = \frac{1}{\sqrt{N}}\sum_k \mathbb{e}^{i \frac{2\mathbb{\pi}}{N} k\mathbb{i}} Y_k\\,.$$

    Calculating gives
    $$S_L[X^\ast_i, X_j] = S_L[Y^\ast_i, Y_j] = \sum_{k=0}^{N-1} \int_{t_0}^{t_1}\mathbb{d} t \left\\{ \frac{1}{2}m \dot Y^\ast_k \dot Y_k - \frac{1}{2}m \omega^2\cdot4\sin^2\frac{2\mathbb{\pi}k}{N} Y^\ast_k Y_k\right\\}\\,.$$
    Using the same transformation to separate the non-dynamic phases, we can arrive at a real action
    $$S_L[y] = \sum_{k=0}^{N-1} \int_{t_0}^{t_1}\mathbb{d} t \left\\{ \frac{1}{2}m \dot y_k^2 - \frac{1}{2}m \omega^2\cdot4\sin^2\frac{2\mathbb{\pi}k}{N} y_k^2\right\\}\\,.$$

    The origional system can then be solved by $N$ independent oscillators
    $$\dot y_k^2 + 4\omega^2\sin^2\frac{2\mathbb{\pi}k}{N} y_k^2 \equiv 4\omega^2\sin^2\frac{2\mathbb{\pi}k}{N} y_{k0}^2\,.$$

    Since the original degrees of freedom are real, the initial conditions of the propagating waves need to satisfy
    $Y_k = Y^*_{-k \mod N}$, see [Wikipedia](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#DFT_of_real_and_purely_imaginary_signals).
    """

    def __init__(
        self,
        omega: float,
        half_initial_conditions: Collection[Mapping[str, complex | float | int]],
        odd_dof: bool,
    ) -> None:
        self.omega = omega
        self.dof = len(half_initial_conditions) + odd_dof
        self.travelling_waves = [
            SimpleComplexHarmonicOscillator(
                dict(
                    omega=2 * omega * np.sin(np.pi * k / self.dof),
                    real=not odd_dof and k == len(half_initial_conditions),
                ),
                dict(x0=ic["y0"], phi=ic["phi"]),
            )
            for k, ic in enumerate(half_initial_conditions)
        ]
        if odd_dof:
            duplicate = self.travelling_waves[-1:0:-1]
        else:
            if self.travelling_waves[-1].initial_condition.x0.imag != 0:
                raise ValueError("The amplitude of wave number N // 2 must be real")
            duplicate = self.travelling_waves[-2:0:-1]
        self.travelling_waves += [
            SimpleComplexHarmonicOscillator(
                dict(omega=-sho.system.omega),
                dict(x0=(ic := sho.initial_condition).x0.conjugate(), phi=-ic.phi),
            )
            for sho in duplicate
        ]

    @cached_property
    def definition(self) -> dict[str, Any]:
        """model params and initial conditions defined as a dictionary."""
        return dict(
            omega=self.omega,
            dof=self.dof,
            travelling_waves=[tw.definition for tw in self.travelling_waves],
        )

    def _z(self, t: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        if is_scalar(t):
            t = np.asarray([t])
        travelling_waves = np.asarray([tw._z(t) for tw in self.travelling_waves])
        original_zs = ifft(travelling_waves, axis=0, norm="ortho")
        return original_zs, travelling_waves

    def _x(self, t: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        original_xs, travelling_waves = self._z(t)

        return np.real(original_xs), travelling_waves

    def __call__(self, t: ArrayLike) -> pd.DataFrame:
        """Generate time series data for the harmonic oscillator chain.

        Returns float(s) representing the displacement at the given time(s).

        :param t: time.
        """
        if is_scalar(t):
            t = np.asarray([t])
        original_xs, travelling_waves = self._x(t)
        data = {
            f"{name}{i}": values
            for name, xs in zip(("x", "y"), (original_xs, travelling_waves))
            for i, values in enumerate(xs)
        }

        return pd.DataFrame(data, index=t)
