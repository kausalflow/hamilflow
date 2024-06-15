import math
from typing import Any, Dict, cast

from pydantic import BaseModel, Field


class HarmonicOscillatorIC(BaseModel):
    """The initial condition for a harmonic oscillator

    :cvar x0: initial displacement
    :cvar v0: initial velocity
    :cvar phi: initial phase
    """

    x0: float = Field(default=1.0)
    v0: float = Field(default=0.0)
    phi: float = Field(default=0.0)


def parse_ic_for_sho(omega: float, **kwargs: Any) -> Dict[str, float]:
    "Support alternative initial conditions"
    match keys := {*kwargs.keys()}:
        case set() if keys <= {"x0", "v0", "phi"}:
            ret = {str(k): float(v) for k, v in kwargs.items()}
        case set() if keys == {"x0", "t0"}:
            ret = dict(x0=float(kwargs["x0"]), v0=0.0, phi=-float(omega * kwargs["t0"]))
        case set() if keys == {"E", "t0"}:
            ene = cast(float, kwargs["E"])
            ret = dict(
                x0=math.sqrt(2 * ene) / omega, v0=0.0, phi=-float(omega * kwargs["t0"])
            )
        case _:
            raise ValueError(
                f"Unsupported variable names as an initial condition: {keys}"
            )
    if phi := ret.get("phi"):
        ret["phi"] = phi % (2 * math.pi)

    return ret
