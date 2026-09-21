"""The answer type. No MLX here, so a client can import it without a GPU."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Decision:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    logits: dict[str, float] = field(repr=False, default_factory=dict)
    disagreement: float | None = None
    input_tokens: int = 0
    seconds: float = 0.0

    def above(self, threshold: float) -> str | None:
        """The choice, or None when it should go to a person instead."""
        return self.choice if self.confidence >= threshold else None
