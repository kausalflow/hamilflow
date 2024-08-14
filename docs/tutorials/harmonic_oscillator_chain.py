# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Harmonic oscillator circle
#
# In this tutorial, we show a few interesting examples of the harmonic oscillator circle.

# %%
import math

import numpy as np
from plotly import express as px

from hamilflow.models.harmonic_oscillator_chain import HarmonicOscillatorsChain

# %% [markdown]
# ## Basic setups

# %%
t = np.linspace(0, 3, 257)
omega = 2 * math.pi
pattern_x = r"x\d+"


# %% [markdown]
# ## Fundamental right-moving wave

# %%
ics = [dict(x0=0, v0=0), dict(amp=(1, 0))] + [dict(amp=(0, 0))] * 5
hoc = HarmonicOscillatorsChain(omega, ics, True)

df_res = hoc(t)
df_x_wide = df_res.loc[:, df_res.columns.str.match(pattern_x)]
px.imshow(df_x_wide, origin="lower", labels={"y": "t"})

# %% [markdown]
# ## Fundamental left-moving wave

# %%
ics = [dict(x0=0, v0=0), dict(amp=(0, 1))] + [dict(amp=(0, 0))] * 5
hoc = HarmonicOscillatorsChain(2 * math.pi, ics, True)

df_res = hoc(t)
df_x_wide = df_res.loc[:, df_res.columns.str.match(pattern_x)]
px.imshow(df_x_wide, origin="lower", labels={"y": "t"})

# %% [markdown]
# ## Faster right-moving wave
# Also known as the first harmonic.

# %%
ics = [dict(x0=0, v0=0), dict(amp=(0, 0)), dict(amp=(1, 0))] + [dict(amp=(0, 0))] * 4
hoc = HarmonicOscillatorsChain(omega, ics, True)

df_res = hoc(t)
df_x_wide = df_res.loc[:, df_res.columns.str.match(pattern_x)]
px.imshow(df_x_wide, origin="lower", labels={"y": "t"})

# %% [markdown]
# ## Fundamental stationary wave, odd dof

# %%
ics = [dict(x0=0, v0=0), dict(amp=(1, 1))] + [dict(amp=(0, 0))] * 5
hoc = HarmonicOscillatorsChain(omega, ics, True)

df_res = hoc(t)
df_x_wide = df_res.loc[:, df_res.columns.str.match(pattern_x)]
px.imshow(df_x_wide, origin="lower", labels={"y": "t"})


# %% [markdown]
# ## Fundamental stationary wave, even dof
# There are stationary nodes at $i = 3, 9$.

# %%
ics = [dict(x0=0, v0=0), dict(amp=(1, 1))] + [dict(amp=(0, 0))] * 5
hoc = HarmonicOscillatorsChain(omega, ics, False)

df_res = hoc(t)
df_x_wide = df_res.loc[:, df_res.columns.str.match(pattern_x)]
px.imshow(df_x_wide, origin="lower", labels={"y": "t"})


# %% [markdown]
# ## Linearly moving chain and fundamental right-moving wave

# %%
ics = [dict(x0=0, v0=2), dict(amp=(1, 0))] + [dict(amp=(0, 0))] * 5
hoc = HarmonicOscillatorsChain(omega, ics, False)

df_res = hoc(t)
df_x_wide = df_res.loc[:, df_res.columns.str.match(pattern_x)]
px.imshow(df_x_wide, origin="lower", labels={"y": "t"})

# %% [markdown]
# ## Mathematical-physical discription
# A one-dimensional circle of $N$ interacting harmonic oscillators can be described by the Lagrangian action
# $$S_L[x_i] = \int_{t_0}^{t_1}\mathbb{d} t \left\\{ \sum_{i=0}^{N-1} \frac{1}{2}m \dot x_i^2 - \frac{1}{2}m\omega^2\left(x_i - x_{i+1}\right)^2 \right\\}\\,,$$
# where $x_N \coloneqq x_0$.
#
# This system can be solved in terms of _travelling waves_, obtained by discrete Fourier transform.
#
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
# We can arrive at an action for the Fourier modes
# $$S_L[Y, Y^*] = \sum_{k=0}^{N-1} \int_{t_0}^{t_1}\mathbb{d} t \left\\{ \frac{1}{2}m \dot Y_k^* Y_k - \frac{1}{2}m \omega^2\cdot4\sin^2\frac{2\mathbb{\pi}k}{N} Y_k^* Y_k\right\\}\\,.$$
#
# The origional system can then be solved by $N$ complex oscillators.
#
# Since the original degrees of freedom are real, the initial conditions of the propagating waves need to satisfy
# $Y_k = Y^*_{-k \mod N}$, see [Wikipedia](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#DFT_of_real_and_purely_imaginary_signals).

# %% [markdown]
# # End of Notebook
