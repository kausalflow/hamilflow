"""Trigonometric functions."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Collection

    from numpy import typing as npt


def acos_with_shift(
    x: "Collection[float] | npt.ArrayLike",
    shift: "Collection[float] | npt.ArrayLike | None" = None,
) -> "npt.ArrayLike":
    """Arccos with shift."""
    x = np.asarray(x)
    value = np.arccos(x)
    shift = np.asarray(shift)
    period_shift = (div := np.floor(shift)) * 2 * np.pi
    remainder = shift - div
    value = np.where(remainder <= 0.5, value, 2 * np.pi - value)  # noqa: PLR2004

    return period_shift + value
