"""Typed decisions from a frozen open model, in one forward pass, on Apple Silicon."""

from .decision import Decision

__all__ = ["Rev", "Client", "Decision"]
__version__ = "0.2.0"


def __getattr__(name):
    # Rev pulls in MLX and Client does not; import each only when asked for.
    if name == "Rev":
        from .decide import Rev
        return Rev
    if name == "Client":
        from .client import Client
        return Client
    raise AttributeError(name)
