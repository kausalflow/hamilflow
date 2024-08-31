"""Main module for a harmonic oscillator chain."""

from functools import cached_property
from typing import Mapping, Sequence, cast

import numpy as np
import pandas as pd
from numpy import typing as npt
from scipy.fft import ifft

from .free_particle import FreeParticle
from .harmonic_oscillator import ComplexSimpleHarmonicOscillator


class HarmonicOscillatorsChain:
    r"""Generate time series data for a coupled harmonic oscillator chain with periodic boundary condition.

    A one-dimensional circle of $N$ interacting harmonic oscillators can be described by the Lagrangian action
    $$S_L[x_i] = \int_{t_0}^{t_1}\mathbb{d} t \left\{ \sum_{i=0}^{N-1} \frac{1}{2}m \dot x_i^2 - \frac{1}{2}m\omega^2\left(x_i - x_{i+1}\right)^2 \right\}\,,$$
    where $x_N \coloneqq x_0$.

    This system can be solved in terms of _travelling waves_, obtained by discrete Fourier transform.

    Since the original degrees of freedom are real, the initial conditions of the propagating waves need to satisfy
    $Y_k = Y^*_{-k \mod N}$, see [Wikipedia](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#DFT_of_real_and_purely_imaginary_signals).

    :param omega: frequence parameter
    :param initial_conditions: a sequence of initial conditions on the Fourier modes.
        The first element in the sequence is that of the zero mode, taking a position and a velocity.
        Rest of the elements are that of the independent travelling waves, taking two amplitudes and two initial phases.
    :param odd_dof: The system will have `2 * len(initial_conditions) + int(odd_dof) - 2` degrees of freedom.
    """

    def __init__(
        self,
        omega: float,
        initial_conditions: Sequence[Mapping[str, float | tuple[float, float]]],
        odd_dof: bool,
    ) -> None:
        self.n_dof = 2 * len(initial_conditions) + odd_dof - 2
        if not odd_dof:
            prefix = "For even degrees of freedom, "
            if self.n_dof == 0:
                raise ValueError(prefix + "at least 1 travelling wave is needed")
            amp = cast(tuple[float, float], initial_conditions[-1]["amp"])
            if amp[0] != amp[1]:
                msg = "k == N // 2 must have equal positive and negative amplitudes."
                raise ValueError(prefix + msg)
        self.omega = omega
        self.odd_dof = odd_dof

        self.free_mode = FreeParticle(cast(Mapping[str, float], initial_conditions[0]))

        self.independent_csho_modes = [
            self._sho_factory(
                k,
                cast(tuple[float, float], ic["amp"]),
                cast(tuple[float, float] | None, ic.get("phi")),
            )
            for k, ic in enumerate(initial_conditions[1:], 1)
        ]

    def _sho_factory(
        self,
        k: int,
        amp: tuple[float, float],
        phi: tuple[float, float] | None = None,
    ) -> ComplexSimpleHarmonicOscillator:
        return ComplexSimpleHarmonicOscillator(
            dict(
                omega=2 * self.omega * np.sin(np.pi * k / self.n_dof),
            ),
            dict(x0=amp) | (dict(phi=phi) if phi else {}),
        )

    @cached_property
    def definition(
        self,
    ) -> dict[
        str,
        float
        | dict[str, dict[str, float | list[float]]]
        | list[dict[str, dict[str, float | tuple[float, float]]]],
    ]:
        """Model params and initial conditions defined as a dictionary."""
        return dict(
            omega=self.omega,
            n_dof=self.n_dof,
            free_mode=self.free_mode.definition,
            independent_csho_modes=[
                rwm.definition for rwm in self.independent_csho_modes
            ],
        )

    def _z(
        self,
        t: "Sequence[float] | npt.ArrayLike",
    ) -> "tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]":
        t = np.asarray(t).reshape(-1)
        all_travelling_waves = [self.free_mode._x(t).reshape(1, -1)]

        if self.independent_csho_modes:
            independent_cshos = np.asarray(
                [o._z(t) for o in self.independent_csho_modes],
            )
            all_travelling_waves.extend(
                (
                    (independent_cshos, independent_cshos[::-1].conj())
                    if self.odd_dof
                    else (
                        independent_cshos[:-1],
                        independent_cshos[[-1]],
                        independent_cshos[-2::-1].conj(),
                    )
                ),
            )

        travelling_waves = np.concatenate(all_travelling_waves)
        original_zs = ifft(travelling_waves, axis=0, norm="ortho")
        return original_zs, travelling_waves

    def _x(
        self,
        t: "Sequence[float] | npt.ArrayLike",
    ) -> "tuple[npt.NDArray[np.float64], npt.NDArray[np.complex64]]":
        original_xs, travelling_waves = self._z(t)

        return np.real(original_xs), travelling_waves

    def __call__(self, t: "Sequence[float] | npt.ArrayLike") -> pd.DataFrame:
        """Generate time series data for the harmonic oscillator chain.

        Returns float(s) representing the displacement at the given time(s).

        :param t: time.
        """
        t = np.asarray(t)
        original_xs, travelling_waves = self._x(t)
        data = {  # type: ignore [var-annotated]
            f"{name}{i}": cast(
                "npt.NDArray[np.float64] | npt.NDArray[np.complex64]",
                values,
            )
            for name, xs in zip(("x", "y"), (original_xs, travelling_waves))
            for i, values in enumerate(xs)  # type: ignore [arg-type]
        }

        return pd.DataFrame(data, index=t)
