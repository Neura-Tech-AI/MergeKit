#!/usr/bin/env python3
"""Build two tiny random Llama checkpoints for offline mergekit experiments.

Hugging Face is not reachable from the sandbox, so real checkpoints cannot be
downloaded. These ~115k-parameter models let every code path in mergekit run
locally: graph construction, lazy loading, merge methods, sharding, tokenizer
copying and model card generation.

Usage:
    python tools/demo/make_tiny_models.py --out /tmp/mk-demo
"""

import argparse
import json
import os
from typing import Dict, Tuple

import torch
from transformers import LlamaConfig, LlamaForCausalLM

# name -> (description, weight offset used to make the two models differ)
VARIANTS: Dict[str, Tuple[str, float]] = {
    "model-a": ("cooking expert", 0.0),
    "model-b": ("coding expert", 0.7),
}

VOCAB_SIZE = 256


def build_tokenizer() -> dict:
    """Minimal but valid tokenizers BPE file covering VOCAB_SIZE tokens."""
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {f"t{i}": i for i in range(VOCAB_SIZE)},
            "merges": [],
        },
    }


def make_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )


def write_model(out_dir: str, name: str, offset: float, seed: int) -> None:
    torch.manual_seed(seed)
    model = LlamaForCausalLM(make_config())
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.02 + offset * 0.01)

    path = os.path.join(out_dir, name)
    model.save_pretrained(path, safe_serialization=True)
    with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(build_tokenizer(), f)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"saved {path} ({n_params:,} parameters)")


def write_configs(out_dir: str) -> None:
    linear = f"""\
models:
  - model: {out_dir}/model-a
    parameters:
      weight: 0.6
  - model: {out_dir}/model-b
    parameters:
      weight: 0.4
merge_method: linear
dtype: float32
"""
    ties = f"""\
models:
  - model: {out_dir}/model-a
    parameters:
      density: 0.7
      weight: 1.0
  - model: {out_dir}/model-b
    parameters:
      density: 0.7
      weight: 0.8
base_model: {out_dir}/model-a
merge_method: ties
parameters:
  normalize: true
dtype: float32
"""
    for fname, body in (("linear.yml", linear), ("ties.yml", ties)):
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(body)
        print(f"wrote {os.path.join(out_dir, fname)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="/tmp/mk-demo", help="output directory (default: /tmp/mk-demo)"
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    for seed, (name, (_desc, offset)) in enumerate(VARIANTS.items()):
        write_model(out_dir, name, offset, seed)

    write_configs(out_dir)
    print("\nNow run:")
    print(f"  mergekit-yaml {out_dir}/linear.yml {out_dir}/out-linear --copy-tokenizer")


if __name__ == "__main__":
    main()
