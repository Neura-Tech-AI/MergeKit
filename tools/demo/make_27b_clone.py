# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Build a shape-faithful *clone* of `sartajbhuvaji/unfuse-Qwen3.8-27B-text`.

The published checkpoint is ~49 GB in bf16, which cannot be downloaded or merged
inside a small sandbox. This script copies its **architecture** exactly and only
shrinks the widths, so a real `mixtureforge-moe` run exercises the same code path:

* 64 layers (the real count)
* the real hybrid pattern -- 48 `linear_attention` + 16 `full_attention`, full
  attention every 4th layer, `full_attention_interval: 4`
* GatedDeltaNet parameters (k/v head counts, conv kernel, A_log, dt_bias)
* `attn_output_gate: true` (q_proj is twice the width of the attention output)
* untied embeddings, partial rotary 0.25, mRoPE
* an `mtp.*` head, stored outside `model.` like the published checkpoint
* optional `model.language_model.` prefix (the VLM-layout variant)

Unlike `make_tiny_qwen35.py` this ships a **real WordLevel tokenizer**, so prompt
encoding is non-empty and `gate_mode: hidden` works end to end.

    python tools/demo/make_27b_clone.py --out /tmp/clone
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoConfig, AutoTokenizer
from transformers.initialization import no_init_weights

# --------------------------------------------------------------------------- #
# architecture: real values from the published config.json, widths shrunk
# --------------------------------------------------------------------------- #

# 48 linear + 16 full, full attention on every 4th layer -- verbatim from the
# real config so the hybrid dispatch code is exercised with the true density.
LAYER_TYPES = [
    "full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)
]

VOCAB_WORDS = [
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "is",
    "are",
    "and",
    "or",
    "not",
    "for",
    "with",
    "that",
    "this",
    "it",
    "as",
    "at",
    "be",
    "by",
    "from",
    "on",
    "was",
    "were",
    "which",
    "what",
    "when",
    "where",
    "who",
    "how",
    "why",
    "can",
    "write",
    "code",
    "function",
    "python",
    "json",
    "parse",
    "error",
    "handle",
    "debug",
    "stack",
    "trace",
    "explain",
    "root",
    "cause",
    "refactor",
    "make",
    "readable",
    "math",
    "prove",
    "square",
    "sqrt",
    "irrational",
    "step",
    "solve",
    "problem",
    "carefully",
    "verify",
    "answer",
    "think",
    "logic",
    "puzzle",
    "before",
    "answering",
    "story",
    "short",
    "lighthouse",
    "keeper",
    "describe",
    "scene",
    "vivid",
    "poetic",
    "language",
    "continue",
    "narrative",
    "engaging",
    "style",
    "recursion",
    "recursive",
    "algorithm",
    "data",
    "structure",
    "list",
    "array",
    "number",
    "numbers",
    "theorem",
    "proof",
    "lemma",
    "therefore",
    "because",
    "assume",
    "suppose",
    "conclude",
    "equals",
    "sum",
    "product",
    "prime",
    "even",
    "odd",
    "integer",
    "real",
    "set",
    "map",
    "sequence",
    "limit",
    "derivative",
    "integral",
    "gradient",
    "loss",
    "model",
    "train",
    "training",
    "merge",
    "router",
    "expert",
    "experts",
    "token",
    "tokens",
    "layer",
    "attention",
    "linear",
    "hybrid",
    "output",
    "input",
    "weight",
    "weights",
    "tensor",
    "shard",
    "config",
    "test",
    "tests",
    "pass",
    "fail",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "hello",
    "world",
    "good",
    "bad",
    "best",
    "worst",
    "fast",
    "slow",
    "big",
    "small",
    "long",
    "short",
    "first",
    "last",
    "next",
    "previous",
    "true",
    "false",
    "class",
    "def",
    "return",
    "import",
    "if",
    "else",
    "while",
    "loop",
    "index",
    "value",
    "key",
    "dict",
    "string",
    "integer",
    "float",
    "bool",
    "none",
    "south",
    "north",
    "east",
    "west",
    "day",
    "night",
    "light",
    "dark",
    "sea",
    "wave",
    "wind",
    "stone",
    "tower",
    "star",
    "sky",
    "river",
    "mountain",
]

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"


def _vocab_size() -> int:
    """Size of the byte-level BPE vocabulary built by `_write_tokenizer`."""
    words = set()
    words.add("<|endoftext|>")
    words.update({c for w in VOCAB_WORDS for c in w})
    words.add("\u0120")
    for w in VOCAB_WORDS:
        words.add("\u0120" + w[0])
        for i in range(1, len(w)):
            words.add("\u0120" + w[: i + 1])
    return len(words)


def _clone_config(vlm_prefix: bool) -> dict:
    """Real architecture, shrunk widths (see module docstring)."""
    return {
        "architectures": ["Qwen3_5ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "bos_token_id": 0,
        "dtype": "float32",
        "eos_token_id": 0,
        "full_attention_interval": 4,
        # head_dim 32 with 4 heads -> q_proj is 2*4*32 = 256 wide (output gate)
        "head_dim": 32,
        "hidden_act": "silu",
        "hidden_size": 128,
        "initializer_range": 0.02,
        # 3 * 128 * 256 per MLP, x 64 layers, x 3 experts -- cheap to merge
        "intermediate_size": 256,
        "layer_types": LAYER_TYPES,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 6,
        "linear_value_head_dim": 32,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 4096,
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": 4,
        "num_hidden_layers": 64,
        "num_key_value_heads": 2,
        "output_gate_type": "swish",
        "pad_token_id": 0,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": _vocab_size(),
    }


def _mtp_tensors(cfg) -> dict:
    """Fake MTP head with the published checkpoint's `mtp.*` naming."""
    hidden, inter = cfg.hidden_size, cfg.intermediate_size
    head_dim = cfg.head_dim
    return {
        "mtp.fc.weight": torch.randn(hidden, 2 * hidden) * 0.02,
        "mtp.pre_fc_norm_embedding.weight": torch.ones(hidden),
        "mtp.pre_fc_norm_hidden.weight": torch.ones(hidden),
        "mtp.layers.0.input_layernorm.weight": torch.ones(hidden),
        "mtp.layers.0.post_attention_layernorm.weight": torch.ones(hidden),
        "mtp.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden) * 0.02,
        "mtp.layers.0.self_attn.k_proj.weight": torch.randn(hidden, hidden) * 0.02,
        "mtp.layers.0.self_attn.v_proj.weight": torch.randn(hidden, hidden) * 0.02,
        "mtp.layers.0.self_attn.o_proj.weight": torch.randn(hidden, hidden) * 0.02,
        "mtp.layers.0.self_attn.q_norm.weight": torch.ones(head_dim),
        "mtp.layers.0.self_attn.k_norm.weight": torch.ones(head_dim),
        "mtp.layers.0.mlp.gate_proj.weight": torch.randn(inter, hidden) * 0.02,
        "mtp.layers.0.mlp.up_proj.weight": torch.randn(inter, hidden) * 0.02,
        "mtp.layers.0.mlp.down_proj.weight": torch.randn(hidden, inter) * 0.02,
        "mtp.norm.weight": torch.ones(hidden),
    }


def _write_tokenizer(path: str) -> None:
    """Emit a *real* Qwen3.5 tokenizer (byte-level BPE), not a stand-in.

    `Qwen3_5Tokenizer` subclasses `TokenizersBackend` and hardcodes a byte-level
    BPE pipeline in `__init__`. A WordLevel `tokenizer.json` therefore loads as
    an all-but-unknown vocabulary and encodes everything to `[]`. Building the
    BPE through the tokenizer class keeps the fixture faithful to the published
    checkpoint and keeps the save/reload round-trip exact.
    """
    from transformers.models.qwen3_5.tokenization_qwen3_5 import Qwen3_5Tokenizer

    vocab: dict[str, int] = {}
    merges: list[tuple[str, str]] = []

    def add(token: str) -> None:
        if token not in vocab:
            vocab[token] = len(vocab)

    add("<|endoftext|>")
    for ch in sorted({c for w in VOCAB_WORDS for c in w}):
        add(ch)
    add("\u0120")  # the byte-level space marker

    for word in VOCAB_WORDS:
        add("\u0120" + word[0])
        merges.append(("\u0120", word[0]))
        for i in range(1, len(word)):
            add("\u0120" + word[: i + 1])
            merges.append(("\u0120" + word[:i], word[i]))

    tokenizer = Qwen3_5Tokenizer(
        vocab=vocab,
        merges=merges,
        pad_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(path)

    with open(
        os.path.join(path, "generation_config.json"), "w", encoding="utf-8"
    ) as fp:
        json.dump({"bos_token_id": 0, "eos_token_id": 0, "pad_token_id": 0}, fp)
    return len(vocab)


def build(
    out_dir: str,
    name: str,
    seed: int,
    vlm_prefix: bool = False,
    with_mtp: bool = True,
    layer_bias: float = 0.0,
) -> str:
    """Create one dense Qwen3.5 clone. `layer_bias` nudges MLP weights per expert."""
    from safetensors.torch import save_file
    from transformers import Qwen3_5ForCausalLM

    torch.manual_seed(seed)
    raw = _clone_config(vlm_prefix)
    model_cfg = AutoConfig.for_model(
        "qwen3_5_text", **{k: v for k, v in raw.items() if k != "model_type"}
    )

    with torch.device("meta"), no_init_weights():
        model = Qwen3_5ForCausalLM(model_cfg)

    state = {}
    for key, tensor in model.state_dict().items():
        value = torch.randn(tensor.shape) * 0.02
        if key.endswith("norm.weight") or "layernorm" in key:
            value = torch.ones(tensor.shape)
        if key.endswith("A_log"):
            value = torch.ones(tensor.shape)
        if "mlp." in key and layer_bias:
            value = value + layer_bias
        state[key] = value

    if with_mtp:
        state.update(_mtp_tensors(model_cfg))

    if vlm_prefix:
        state = {
            f"model.language_model.{k[len('model.') :] if k.startswith('model.') else k}": v
            for k, v in state.items()
        }

    path = os.path.join(out_dir, name)
    os.makedirs(path, exist_ok=True)
    save_file(state, os.path.join(path, "model.safetensors"))

    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fp:
        json.dump(raw, fp, indent=2)

    _write_tokenizer(path)

    # a chat template + aux file, so tagalong copying has something to carry
    with open(os.path.join(path, "chat_template.jinja"), "w", encoding="utf-8") as fp:
        fp.write("{{ messages }}\n")

    total = sum(float(v.numel()) for v in state.values())
    print(f"saved {path}: {len(state)} tensors, {total / 1e6:.2f}M params")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/clone")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="clone")
    parser.add_argument("--vlm-prefix", action="store_true")
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--layer-bias", type=float, default=0.0)
    args = parser.parse_args()
    build(
        args.out,
        args.name,
        args.seed,
        vlm_prefix=args.vlm_prefix,
        with_mtp=not args.no_mtp,
        layer_bias=args.layer_bias,
    )
    AutoTokenizer.from_pretrained(os.path.join(args.out, args.name))


if __name__ == "__main__":
    main()
