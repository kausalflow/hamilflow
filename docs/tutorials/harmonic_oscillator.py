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
# # Harmonic Oscillators
#
# In this tutorial, we demo how to generate data of harmonic oscillators.

# %%
import pandas as pd
import plotly.express as px

from hamiltonian_flow.models.harmonic_oscillator import HarmonicOscillator

# %%
n_periods = 3
n_samples_per_period = 200

# %% [markdown]
# ## Simple Harmonic Oscillator

# %%
sho_omega = 0.5

sho = HarmonicOscillator(system={"omega": sho_omega})

# %%
df_sho = sho(n_periods=n_periods, n_samples_per_period=n_samples_per_period)
df_sho.head()

# %%
px.line(
    df_sho,
    x="t",
    y="x",
    title=rf"Simple Harmonic Oscillator (omega = {sho_omega})",
    labels={
        "x": r"Displacement $x(t)$",
        "t": r"$t$",
    },
)

# %% [markdown]
# ## Damped Harmonic Oscillator

# %%
dho_systems = {
    "Underdamped": {"omega": 0.5, "zeta": 0.2},
    "Critical Damped": {"omega": 0.5, "zeta": 1},
    "Overdamped": {
        "omega": 0.5,
        "zeta": 1.2,
    },
}

dfs_dho = []

for s_name, s in dho_systems.items():

    dfs_dho.append(
        HarmonicOscillator(system=s)(
            n_periods=n_periods, n_samples_per_period=n_samples_per_period
        ).assign(system=rf"{s_name} (omega = {s.get('omega')}, zeta = {s.get('zeta')})")
    )

fig = px.line(
    pd.concat(dfs_dho),
    x="t",
    y="x",
    color="system",
    title=rf"Damped Harmonic Oscillator",
    labels={
        "x": r"Displacement $x(t)$",
        "t": r"$t$",
    },
)
fig.update_layout(legend={"yanchor": "top", "y": -0.2, "xanchor": "left", "x": 0})

# %%
