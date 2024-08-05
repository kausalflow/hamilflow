from functools import cached_property
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy as sp
from pydantic import BaseModel, Field, computed_field, field_validator


PLANCK_CONST = 6.626e-34
PI = 3.142


class FiniteBox(BaseModel):
    r"""Definition of the potential of the one dimensional box

    For consistency, we always use $\mathbf x$ for displacement,
    $L$ for box size.

    The potential distribution we use is

    $$
    V(x)=\left\{\begin{array}{ll}
    -V_{0} & |x|<L \\
    0 & |x|>L
    \end{array}\right.
    $$

    :cvar length: length of the box
    :cvar p_height: potential well height
    """

    length: float = Field(ge=0)
    p_height: float = Field(g=0, default=1.0)


class QuantumParticle(BaseModel):
    r"""Definition of the quantum particle

    :cvar energy: energy of the particle
    :cvar mass: weight of the particle
    """

    energy: float = Field(l=0, ge=-FiniteBox.p_height)
    mass: float = Field(ge=0, default=1e-5)


class WaveFunctionConst(BaseModel):
    r"""Schrodinger equation simplification terms"""

    @property
    def h_bar(self) -> float:
        """The hbar simplification
        """
        return PLANCK_CONST / (2 * PI)

    @computed_field  # type: ignore[misc]
    @cached_property
    def calculate_alpha(self) -> float:
        """Simplification term alpha

        $$
        \alpha=\sqrt{\frac{2 m}{\hbar^{2}}\left(V_{0}-|E|\right)}
        $$
        """
        return (
            np.sqrt(
                2 * QuantumParticle.mass * (FiniteBox.p_height - np.abs(QuantumParticle.energy))
                / (self.h_bar ** 2)
            )
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def calculate_beta(self) -> float:
        """Simplification term beta

        $$
        \beta=\sqrt{\frac{2 m}{\hbar^{2}}|E|}
        $$
        """
        return (
            np.sqrt(
                2 * QuantumParticle.mass * np.abs(QuantumParticle.energy) / (self.h_bar ** 2)
            )
        )


class ParticleInBox:
    r"""Definition of the particle in a box quantum system

    For consistency, we always use
    $\mathbf x$ for displacement, $E$ for particle energy,
    $V$ for barrier energy, and $L$ for box size.
    The one-dimensional Schrödinger equation we are using is

    $$
    \begin{align}
    -{\frac{\hbar^2}{2m}} {\frac{\partial^2 \psi}{\partial x^2}} + V(r) \psi = E \psi \label{eq1}
    \end{align}
    $$

    And the even wave functions we are using are

    $$
    \psi^{F}(x)=\left\{\begin{array}{ll}
    A \cos (\alpha \alpha) & 0<x<L \\
    C e^{-\beta \hbar} & x>L
    \end{array}\right.
    $$

    The odd wave functions we are using are

    $$
    \psi^{o}(x)=\left\{\begin{array}{ll}
    A \sin (\alpha \alpha) & 0<x<L \\
    C e^{-\beta_{x}} & x>L
    \end{array}\right.
    $$

    References:

    1. Particle in a box and tunneling. [cited 26 Mar 2024].
        Available: https://chem.libretexts.org/Courses/University_of_California_Davis/UCD_Chem_110A%3A_Physical_Chemistry__I/UCD_Chem_110A%3A_Physical_Chemistry_I_(Koski)/Text/03%3A_The_Schrodinger_Equation/3.09%3A_Particle_in_a_Finite_Box_and_Tunneling_(optional)
    2. Contributors to Wikimedia projects. Particle in a box.
        In: Wikipedia [Internet]. 22 Jan 2024 [cited 26 Mar 2024].
        Available: https://en.wikipedia.org/wiki/Particle_in_a_box

    :param finite_box: the finite box definition
    :param quantum_particle: the particle in the box
    """

    def __init__(
        self,
        finite_box: Dict[str, float],
        quantum_particle: Dict[str, float],
    ):
        self.finite_box = FiniteBox.model_validate(finite_box)
        self.quantum_particle = QuantumParticle.model_validate(quantum_particle)

    def __call__(self) -> pd.DataFrame:
        pass

