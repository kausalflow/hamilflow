from functools import partial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Collection

    from numpy import typing as npt


def acos_with_shift(
    x: "Collection[float] | npt.ArrayLike",
    shift: "Collection[float] | npt.ArrayLike | None" = None,
) -> "npt.ArrayLike":
    x = np.array(x, copy=False)
    value = np.arccos(x)
    shift = np.array(shift, copy=False)
    period_shift = (div := np.floor(shift)) * 2 * np.pi
    remainder = shift - div
    value = np.where(remainder <= 0.5, value, 2 * np.pi - value)

    return period_shift + value
