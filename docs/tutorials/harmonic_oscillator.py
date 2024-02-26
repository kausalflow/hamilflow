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
import matplotlib.pyplot as plt

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
fig, ax = plt.subplots(figsize=(10, 6.18))

df_sho.plot(x="t", y="x", marker=".", ax=ax)

ax.set_title(rf"Simple Harmonic Oscillator ($\omega = {sho_omega}$)")
ax.set_ylabel(r"Displacement $x(t)$")
ax.set_xlabel(r"$t$")

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


for s_name, s in dho_systems.items():
    fig, ax = plt.subplots(figsize=(10, 6.18))

    HarmonicOscillator(system=s)(
        n_periods=n_periods, n_samples_per_period=n_samples_per_period
    ).plot(x="t", y="x", marker=".", ax=ax)

    ax.set_title(
        rf"{s_name} Harmonic Oscillator ($\omega = {s.get('omega')}$, $\zeta = {s.get('zeta')}$)"
    )
    ax.set_ylabel(r"Displacement $x(t)$")
    ax.set_xlabel(r"$t$")
    plt.show()
