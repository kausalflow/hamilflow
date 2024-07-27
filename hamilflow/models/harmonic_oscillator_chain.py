from functools import cached_property
from typing import Mapping, Sequence, cast

import numpy as np
import pandas as pd
from scipy.fft import ifft

from .free_particle import FreeParticle
from .harmonic_oscillator import ComplexSimpleHarmonicOscillator


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
        omega: float | int,
        initial_conditions: Sequence[
            Mapping[str, float | int | tuple[float | int, float | int]]
        ],
        odd_dof: bool,
    ) -> None:
        self.omega = omega
        self.n_independant_csho_dof = len(initial_conditions) - 1
        self.odd_dof = odd_dof

        self.free_mode = FreeParticle(
            cast(Mapping[str, float | int], initial_conditions[0])
        )

        r_wave_modes_ic = initial_conditions[1:]
        self.independent_csho_modes = [
            self._sho_factory(
                k,
                cast(tuple[float | int, float | int], ic["amp"]),
                cast(tuple[float | int, float | int] | None, ic.get("phi")),
            )
            for k, ic in enumerate(r_wave_modes_ic, 1)
        ]

    def _sho_factory(
        self,
        k: int,
        amp: tuple[float | int, float | int],
        phi: tuple[float | int, float | int] | None = None,
    ) -> ComplexSimpleHarmonicOscillator:
        return ComplexSimpleHarmonicOscillator(
            dict(
                omega=2 * self.omega * np.sin(np.pi * k / self.n_dof),
            ),
            dict(x0=amp) | (dict(phi=phi) if phi else {}),
        )

    @cached_property
    def n_dof(self) -> int:
        return self.n_independant_csho_dof * 2 + self.odd_dof

    @cached_property
    def definition(
        self,
    ) -> dict[
        str,
        float
        | int
        | dict[str, dict[str, int | float | list[int | float]]]
        | list[dict[str, dict[str, float | int | tuple[float | int, float | int]]]],
    ]:
        """model params and initial conditions defined as a dictionary."""
        return dict(
            omega=self.omega,
            n_dof=self.n_dof,
            free_mode=self.free_mode.definition,
            independent_csho_modes=[
                rwm.definition for rwm in self.independent_csho_modes
            ],
        )

    def _z(
        self, t: float | int | Sequence[float | int]
    ) -> tuple[np.ndarray, np.ndarray]:
        t = np.array(t, copy=False).reshape(-1)
        all_travelling_waves = [self.free_mode._x(t).reshape(1, -1)]

        if self.independent_csho_modes:
            independent_cshos = np.array(
                [o._z(t) for o in self.independent_csho_modes], copy=False
            )
            all_travelling_waves.extend(
                (independent_cshos, independent_cshos[::-1].conj())
                if self.odd_dof
                else (
                    independent_cshos[:-1],
                    independent_cshos[[-1]],
                    independent_cshos[-1::-1].conj(),
                )
            )

        travelling_waves = np.concatenate(all_travelling_waves)
        original_zs = ifft(travelling_waves, axis=0, norm="ortho")
        return original_zs, travelling_waves

    def _x(
        self, t: float | int | Sequence[float | int]
    ) -> tuple[np.ndarray, np.ndarray]:
        original_xs, travelling_waves = self._z(t)

        return np.real(original_xs), travelling_waves

    def __call__(self, t: float | int | Sequence[float | int]) -> pd.DataFrame:
        """Generate time series data for the harmonic oscillator chain.

        Returns float(s) representing the displacement at the given time(s).

        :param t: time.
        """
        original_xs, travelling_waves = self._x(t)
        data = {
            f"{name}{i}": values
            for name, xs in zip(("x", "y"), (original_xs, travelling_waves))
            for i, values in enumerate(xs)
        }

        return pd.DataFrame(data, index=t)
