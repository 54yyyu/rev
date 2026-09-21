"""One forward pass, one distribution.

Nothing is generated. The prompt stops where the model would emit its answer,
a single forward pass produces the next-token logits, and the logits at the
declared answer slots are softmaxed among themselves.

That last step is a conditional renormalisation: the model also assigns mass to
every other token in the vocabulary, so the result ranks the options against
each other and is not a calibrated probability until `calibrate` has fitted one.
"""

from __future__ import annotations

import numpy as np


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64) / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def confidence(probabilities: np.ndarray) -> float:
    return float(np.max(probabilities))


def disagreement(branches: list[np.ndarray]) -> float:
    """Mean total-variation distance between branches and their mean.

    0 when every option order agrees, 1 when they scatter completely. Measured
    here as a useful review flag on easy work (96% agreement, and the 4% that
    disagree are half wrong) and as no help at all on hard work, where even the
    agreeing half is barely above chance.
    """
    if len(branches) < 2:
        return 0.0
    m = np.vstack(branches)
    mean = m.mean(axis=0)
    return float(0.5 * np.abs(m - mean).sum(axis=1).mean())
