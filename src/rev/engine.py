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
    staged = Path(tempfile.mkdtemp(prefix="rev-"))
    for item in path.iterdir():
        if item.name == "config.json":
            continue
        (staged / item.name).symlink_to(item.resolve())
    config["model_type"] = ALIASES[kind]
    (staged / "config.json").write_text(json.dumps(config, indent=1))
    return staged


DEFAULT_CACHE_LIMIT_MIB = 256

# Architectures whose logits are exactly head(backbone(x)) with nothing after
# the head, so the head can be applied to the last position alone.
LAST_POSITION_SAFE = {"qwen3_5", "qwen3_5_text", "qwen3", "qwen2", "llama"}


def last_position_head(net):
    """(backbone, head) when the output head can be applied to one position.

    A full forward applies the vocabulary projection to every position: on a
    3,000-token prompt that is 3,000 x 248k logits, about 1.5 GB and most of
    the arithmetic, to read one row. Returns None for architectures not known
    to be safe, which then take the full forward.
    """
    lm = getattr(net, "language_model", net)
    body = getattr(lm, "model", None)
    args = getattr(lm, "args", None)
    if body is None or getattr(args, "model_type", None) not in LAST_POSITION_SAFE:
        return None
    if getattr(args, "tie_word_embeddings", False):
        return body, body.embed_tokens.as_linear
    head = getattr(lm, "lm_head", None)
    return (body, head) if head is not None else None


def load(model: str, bits: int | None = 8, group_size: int = 64,
         cache_limit_mib: int = DEFAULT_CACHE_LIMIT_MIB):
    """Returns (model, tokenizer). `bits=None` keeps the checkpoint's precision.

    The allocator cache is bounded. MLX otherwise keeps nearly all of system RAM
    across variable-length prompts, which crowds out anything else on the
    machine - and, because allocation state steers kernel selection, changes the
    last bits of the logits on long inputs. Bounding it is also what makes a run
    reproducible against the reference implementation.
    """
    from mlx_lm import load as mlx_load
    from huggingface_hub import snapshot_download

    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(cache_limit_mib * 1024 * 1024)
    path = Path(model).expanduser()
    if not path.is_dir():
        path = Path(snapshot_download(model))
    path = _aliased_copy(path)

    net, tokenizer = mlx_load(str(path))
    if bits is not None:
        nn.quantize(net, group_size=group_size, bits=bits, mode="affine")
    net.eval()
    mx.eval(net.parameters())
    mx.synchronize()
    return net, tokenizer
