"""Loading a model through MLX, including checkpoints MLX-LM does not name.

In-memory quantization matters more than any prompt choice measured here:
4-bit costs about nine points of accuracy against 8-bit on hard questions and
saves nothing in latency, so 8 is the default and 4 is a deliberate choice.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

ALIASES = {"qwen3_5_text": "qwen3_5"}


def _aliased_copy(path: Path) -> Path:
    """A directory whose config names a model type MLX-LM implements.

    Some fine-tunes publish the same architecture under a different
    `model_type`. Weights are linked, not copied.
    """
    config = json.loads((path / "config.json").read_text())
    kind = config.get("model_type")
    if kind not in ALIASES:
        return path
    staged = Path(tempfile.mkdtemp(prefix="hinge-"))
    for item in path.iterdir():
        if item.name == "config.json":
            continue
        (staged / item.name).symlink_to(item.resolve())
    config["model_type"] = ALIASES[kind]
    (staged / "config.json").write_text(json.dumps(config, indent=1))
    return staged


def load(model: str, bits: int | None = 8, group_size: int = 64):
    """Returns (model, tokenizer). `bits=None` keeps the checkpoint's precision."""
    from mlx_lm import load as mlx_load
    from huggingface_hub import snapshot_download

    path = Path(model).expanduser()
    if not path.is_dir():
        path = Path(snapshot_download(model))
    path = _aliased_copy(path)

    net, tokenizer = mlx_load(str(path))
    if bits is not None:
        nn.quantize(net, group_size=group_size, bits=bits)
        mx.eval(net.parameters())
    return net, tokenizer
