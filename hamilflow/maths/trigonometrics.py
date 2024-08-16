from functools import partial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy import typing as npt


def acos_with_shift(x: "npt.ArrayLike", shift: "npt.ArrayLike") -> "npt.ArrayLike":
    x, shift = map(partial(np.array, copy=False), [x, shift])
    p_shift = (div := np.floor(shift)) * 2 * np.pi
    remainder = shift - div
    p_value = np.arccos(x)
    return p_shift + np.where(remainder <= 0.5, p_value, 2 * np.pi - p_value)
