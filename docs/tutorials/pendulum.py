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
# # Pendulum

# ## Mathematical description
# ### Lagrangian action
# We describe a generic pendulum system by the Lagrangian action
# $$
# S_L[\theta] \equiv \int_{0}^{t_0} \mathbb{d}t\,L(\theta, \dot\theta)
# \eqqcolon I \int_{0}^{t_0} \mathbb{d}t
# \left\{\frac{1}{2} \dot\theta^2 + \omega_0^2 \cos\theta \right\}\,,
# $$
# where $L$ is the Lagrangian;
# $\theta$ is the angular displacement from the vertical to the
# pendulum as the generalised position;
# $I$ is the _inertia parameter_,
# $\omega_0$ the _frequency parameter_, and finally
# $U \coloneqq I\omega_0^2$ is the _potential parameter_.
#
# This setup contains both the single and the physical pendula. For a single
# pendulum,
# $$
# I = m l^2\,,\qquad U = mgl\,,
# $$
# where $m$ is the mass of the pendulum, $l$ is the length of the rod or cord,
# and $g$ is the gravitational acceleration.
#
# #### Integral of motion
# The Lagrangian action does not contain time $t$ explicitly.
# As a result, the system is invariant under a variation of time,
# or $\mathbb{\delta}S / \mathbb{\delta}{t} = 0$.
# This gives an integral of motion
# $$
# \dot\theta\frac{\partial L}{\partial \dot\theta} - L
# \equiv E \eqqcolon U \cos\theta_0\,.
# $$
# Substitution gives
# $$
# \left(\frac{\mathbb{d}t}{\mathbb{d}\theta}\right)^2 = \frac{1}{2\omega_0^2}
# \frac{1}{\cos\theta - \cos\theta_0}\,.
# $$
# #### Coordinate transformation
# For convenience, iIntroduce the coordinate $u$ and the parameter $k$
# $$
# \sin u \coloneqq \frac{\sin\frac{\theta}{2}}{k}\,,\qquad
# k \coloneqq \sin\frac{\theta_0}{2} \in [-1, 1]\,.
# $$
# One arrives at
# $$
# \left(\frac{\mathbb{d}t}{\mathbb{d}u}\right)^2 = \frac{1}{\omega_0^2}
# \frac{1}{1-k^2\sin^2 u}\,.
# $$
# The square root of the second factor on the right-hand side makes an elliptic
# integral.

# %% [markdown]
# End of Notebook
