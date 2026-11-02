"""Fake downloaded models for tests that ask the catalogue what a model is.

The catalogue (local_llm/catalog.py) has no built-in model data any more: size,
context, thinking, tool format, vision and the draft sibling are read from the
model's own files in the Hugging Face cache, and estimated from the repo name
before it is downloaded. These helpers lay a repo out the way huggingface_hub
does (models--org--name/refs/main and snapshots/<revision>/...) in a temporary
cache, so a test can say "this model is on disk and looks like this". Weight
files are sparse, so a "4.3GB" model costs no disk.

Not a pytest fixture: the catalogue tests also run standalone (python
tests/test_catalog.py), so the cache is a context manager.
"""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

GB = 1024 ** 3

# Chat templates reduced to the markers the catalogue reads.
QWEN3_TEMPLATE = ("{%- if enable_thinking is defined and enable_thinking is false %}"
                  "<think>\n\n</think>{%- endif %}<tool_call>{{ tool }}</tool_call>")
QWEN25_TEMPLATE = "<|im_start|>system{{ messages }}<tool_call>{{ tool }}</tool_call>"
LLAMA_TEMPLATE = "<|start_header_id|>{{ role }}<|end_header_id|><|python_tag|>{{ call }}"


@contextmanager
def fake_hf_cache():
    """An empty Hugging Face cache as HF_HUB_CACHE while the block runs; yields its path."""
    old = os.environ.get("HF_HUB_CACHE")
    path = Path(tempfile.mkdtemp(prefix="llm-test-hub-"))
    os.environ["HF_HUB_CACHE"] = str(path)
    try:
        yield path
    finally:
        if old is None:
            os.environ.pop("HF_HUB_CACHE", None)
        else:
            os.environ["HF_HUB_CACHE"] = old


def fake_model(cache: Path, repo_id: str, *, size_gb: float = 1.0, model_type: str = "qwen2",
               vocab_size: int = 151936, context: int = 32768, bits: int | None = 4,
               template: str | None = QWEN25_TEMPLATE, vision: bool = False,
               tokenizer: bool = True, weights: bool = True,
               revision: str = "0000000000000000000000000000000000000001") -> Path:
    """Lay out one downloaded repo in `cache`; returns its snapshot folder.

    tokenizer=False or weights=False leaves the half-finished download that
    downloaded_models() must not list. template=None ships no chat template.
    """
    repo = cache / ("models--" + repo_id.replace("/", "--"))
    snap = repo / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs" / "main").write_text(revision, encoding="utf-8")
    config: dict = {"model_type": model_type, "vocab_size": vocab_size,
                    "max_position_embeddings": context}
    if bits:
        config["quantization"] = {"bits": bits, "group_size": 64}
    if vision:
        config["vision_config"] = {"model_type": f"{model_type}_vision"}
    (snap / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if tokenizer:
        tok = {"chat_template": template} if template is not None else {}
        (snap / "tokenizer_config.json").write_text(json.dumps(tok), encoding="utf-8")
    if weights:
        with open(snap / "model.safetensors", "wb") as handle:
            handle.truncate(int(size_gb * GB))
    return snap
