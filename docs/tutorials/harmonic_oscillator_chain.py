# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # One-dimensional harmonic oscillator chain
# We have worked out the case with periodic boundary condition.

# %%
import math

import numpy as np
from plotly import express as px

from hamilflow.models.d1.harmonic_oscillator_chain import HarmonicOscillatorsChain

hoc = HarmonicOscillatorsChain(
    2 * math.pi,
    [
        dict(y0=0, phi=0),
        dict(y0=complex(1, 1), phi=math.pi / 4),
        dict(y0=complex(0, 0), phi=np.pi / 6),
    ],
    True,
)

print(hoc.definition)

df_res = hoc(np.linspace(0, 4, 257))
df_x_long = (
    df_res[["x0", "x1", "x2", "x3", "x4"]]
    .rename(columns={f"x{i}": i for i in range(5)})
    .melt(var_name="x_index", value_name="displacement", ignore_index=False)
    .rename_axis(index="t")
    .reset_index(drop=False)
)

px.line(df_x_long, x="displacement", y="t", color="x_index")


# %% [markdown]
# ## Mathematical-physical discription
# A one-dimensional circle of $N$ interacting harmonic oscillators can be described by the Lagrangian action
# $$S_L[x_i] = \int_{t_0}^{t_1}\mathbb{d} t \left\\{ \sum_{i=0}^{N-1} \frac{1}{2}m \dot x_i^2 - \frac{1}{2}m\omega^2\left(x_i - x_{i+1}\right)^2 \right\\}\\,,$$
# where $x_N \coloneqq x_0$.
#
# This system can be solved in terms of _travelling waves_, obtained by discrete Fourier transform.
#
# We can complexify the system
# $$S_L[x_i] = S_L[x_i, \phi_j] \equiv S_L[X^\ast_i, X_j] = \int_{t_0}^{t_1}\mathbb{d} t \left\\{ \frac{1}{2}m \dot X^\ast_i \delta_{ij} \dot X_j - \frac{1}{2}m X^\ast_i A_{ij} X_j\right\\}\\,,$$
# where $A_{ij} / \omega^2$ is equal to $(-2)$ if $i=j$, $1$ if $|i-j|=1$ or $|i-j|=N$, and $0$ otherwise;
# $X_i \coloneqq x_i \mathbb{e}^{-\phi_i}$, $X^\ast_i \coloneqq x_i \mathbb{e}^{+\phi_i}$.
#
# $A_{ij}$ can be diagonalised by the inverse discrete Fourier transform
# $$X_i = (F^{-1})_{ik} Y_k = \frac{1}{\sqrt{N}}\sum_k \mathbb{e}^{i \frac{2\mathbb{\pi}}{N} k\mathbb{i}} Y_k\\,.$$
#
# Calculating gives
# $$S_L[X^\ast_i, X_j] = S_L[Y^\ast_i, Y_j] = \sum_{k=0}^{N-1} \int_{t_0}^{t_1}\mathbb{d} t \left\\{ \frac{1}{2}m \dot Y^\ast_k \dot Y_k - \frac{1}{2}m \omega^2\cdot4\sin^2\frac{2\mathbb{\pi}k}{N} Y^\ast_k Y_k\right\\}\\,.$$
# Using the same transformation to separate the non-dynamic phases, we can arrive at a real action
# $$S_L[y] = \sum_{k=0}^{N-1} \int_{t_0}^{t_1}\mathbb{d} t \left\\{ \frac{1}{2}m \dot y_k^2 - \frac{1}{2}m \omega^2\cdot4\sin^2\frac{2\mathbb{\pi}k}{N} y_k^2\right\\}\\,.$$
#
# The origional system can then be solved by $N$ independent oscillators
# $$\dot y_k^2 + 4\omega^2\sin^2\frac{2\mathbb{\pi}k}{N} y_k^2 \equiv 4\omega^2\sin^2\frac{2\mathbb{\pi}k}{N} y_{k0}^2\,.$$
#
# Since the original degrees of freedom are real, the initial conditions of the propagating waves need to satisfy
# $Y_k = Y^*_{-k \mod N}$, see [Wikipedia](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#DFT_of_real_and_purely_imaginary_signals).

# %% [markdown]
# # End of Notebook
