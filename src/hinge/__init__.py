"""Typed decisions from a frozen open model, in one forward pass, on Apple Silicon."""

from .decide import Decision, Hinge

__all__ = ["Hinge", "Decision"]
__version__ = "0.1.0"
