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
# # Brownian Motion
#
# Brownian motion describes motion of small particles with stochastic forces applied to them. The math of Brownian motion can be modeled with Wiener process. In this tutorial, we take a simple form of the model and treat the stochastic forces as Gaussian.
#
# For consistency, we always use $\mathbf x$ for displacement, and $t$ for steps. The model we are using is
#
# $$
# \begin{align}
# \mathbf x(t + \mathrm dt) &= \mathbf x(t) + \mathcal{N}(\mu=0, \sigma=\sigma \sqrt{\mathrm d t})
# \end{align}
# $$
#
#
# Read more:
#
# 1. Brownian motion and random walks. [cited 13 Mar 2024]. Available: https://web.mit.edu/8.334/www/grades/projects/projects17/OscarMickelin/brownian.html
# 2. Contributors to Wikimedia projects. Brownian motion. In: Wikipedia [Internet]. 22 Jan 2024 [cited 13 Mar 2024]. Available: https://en.wikipedia.org/wiki/Brownian_motion
#
#
# `hamilflow` implemented a Brownian motion model called `BrownianMotion`.

# %%
import plotly.express as px

from hamilflow.models.brownian_motion import BrownianMotion

# %% [markdown]
# ## 1D Brownian Motion

# %%
bm_1d = BrownianMotion(
    system={
        "sigma": 1,
        "delta_t": 1,
    },
    initial_condition={"x0": 0},
)

# %% [markdown]
# Call the model to generate 1000 steps.

# %%
df_1d = bm_1d(n_steps=1000)

# %%
px.line(df_1d, x="t", y="y_0")

# %% [markdown]
# ## 2D Brownian Motion
#
# Our `BrownianMotion` model calculates the dimension of the space based on the dimension of the initial condition $x_0$. To create a 2D Brownian motion model, we need the initial condition to be length 2.

# %%
bm_2d = BrownianMotion(
    system={
        "sigma": 1,
        "delta_t": 1,
    },
    initial_condition={"x0": [0, 0]},
)

# %% [markdown]
# We call the model to generate 1000 steps.

# %%
df_2d = bm_2d(n_steps=500)

# %%
(
    px.scatter(df_2d, x="y_0", y="y_1", color="t")
    .update_traces(
        mode="lines+markers",
        marker=dict(
            size=2.5,
        ),
        line=dict(width=1),
    )
    .update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
)

# %%
