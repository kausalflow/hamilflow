# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Complex Harmonic Oscillator
#
# In this tutorial, we show a few interesting examples of the complex simple harmonic oscillator.

# %%
import math

import numpy as np
from plotly import express as px

from hamilflow.models.harmonic_oscillator import ComplexSimpleHarmonicOscillator

# %% [markdown]
# ## Basic setups

# %%
t = np.linspace(0, 3, 257)
system_specs = dict(omega=2 * math.pi)

# %% [markdown]
# ## Positive-frequency, circular-polarised mode
# Also known as the left-rotating mode.

# %%
csho = ComplexSimpleHarmonicOscillator(system_specs, initial_condition=dict(x0=(1, 0)))

df = csho(t)

arr_z = df["z"].to_numpy(copy=False)

px.line_3d(x=arr_z.real, y=arr_z.imag, z=t, labels=dict(x="real", y="imag", z="time"))

# %% [markdown]
# ## Negative-frequency, circular-polarised mode
# Also known as the right-rotating mode.

# %%
csho = ComplexSimpleHarmonicOscillator(system_specs, initial_condition=dict(x0=(0, 1)))

df = csho(t)

arr_z = df["z"].to_numpy(copy=False)

px.line_3d(x=arr_z.real, y=arr_z.imag, z=t, labels=dict(x="real", y="imag", z="time"))

# %% [markdown]
# ## Positive-frequency, elliptic-polarised mode

# %%
csho = ComplexSimpleHarmonicOscillator(
    system_specs,
    initial_condition=dict(x0=(math.cos(math.pi / 12), math.sin(math.pi / 12))),
)

df = csho(t)

arr_z = df["z"].to_numpy(copy=False)

px.line_3d(x=arr_z.real, y=arr_z.imag, z=t, labels=dict(x="real", y="imag", z="time"))

# %% [markdown]
# # End of Notebook
