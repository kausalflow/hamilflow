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
