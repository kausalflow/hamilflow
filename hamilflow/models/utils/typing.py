"""Typing."""

from collections.abc import Sequence
from typing import TypeVar

from numpy.typing import ArrayLike

TypeTime = TypeVar("TypeTime", Sequence[float], Sequence[int], ArrayLike)
