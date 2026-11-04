#!/usr/bin/env python3
"""Build tiny random Qwen3.5 (hybrid attention) checkpoints for offline tests.

Hugging Face is unreachable from this sandbox, so the real
``sartajbhuvaji/unfuse-Qwen3.8-27B-text``-style checkpoint cannot be downloaded.
These tiny checkpoints reproduce the *layout* that matters for MoE conversion:

* hybrid ``layer_types`` (linear attention + every 4th layer full attention),
* ``mtp.*`` multi-token-prediction tensors that the CausalLM class ignores,
* an optional ``model.language_model.*`` VLM prefix.

Usage:
    python tools/demo/make_tiny_qwen35.py --out /tmp/q35
"""

import argparse
import json
import os

import torch
from transformers import AutoConfig
from transformers.initialization import no_init_weights
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM

BASE_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "bos_token_id": 8,
    "dtype": "float32",
    "eos_token_id": 8,
    "full_attention_interval": 4,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 512,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 64,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 8,
    "linear_value_head_dim": 64,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 512,
    "model_type": "qwen3_5_text",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
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
    "vocab_size": 1024,
    "architectures": ["Qwen3_5ForCausalLM"],
}


def mtp_tensors(cfg) -> dict:
    """Fake MTP head matching the published checkpoint's `mtp.*` naming."""
    hidden = cfg.hidden_size
    inter = cfg.intermediate_size
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
        "mtp.layers.0.self_attn.q_norm.weight": torch.ones(64),
        "mtp.layers.0.self_attn.k_norm.weight": torch.ones(64),
        "mtp.layers.0.mlp.gate_proj.weight": torch.randn(inter, hidden) * 0.02,
        "mtp.layers.0.mlp.up_proj.weight": torch.randn(inter, hidden) * 0.02,
        "mtp.layers.0.mlp.down_proj.weight": torch.randn(hidden, inter) * 0.02,
        "mtp.norm.weight": torch.ones(hidden),
    }


def build(out_dir: str, name: str, seed: int, vlm_prefix: bool, with_mtp: bool) -> str:
    torch.manual_seed(seed)
    model_cfg = AutoConfig.for_model(
        "qwen3_5_text",
        **{
            k: v
            for k, v in BASE_CONFIG.items()
            if k not in ("model_type", "architectures")
        },
    )

    with torch.device("meta"), no_init_weights():
        model = Qwen3_5ForCausalLM(model_cfg)

    state = {}
    for key, tensor in model.state_dict().items():
        state[key] = torch.randn(tensor.shape) * 0.02
        if key.endswith("norm.weight") or "layernorm" in key:
            state[key] = torch.ones(tensor.shape)
        if key.endswith("A_log"):
            state[key] = torch.ones(tensor.shape)

    if with_mtp:
        for key, tensor in mtp_tensors(model_cfg).items():
            state[key] = tensor

    if vlm_prefix:
        inner = {}
        for key, value in state.items():
            bare = key[len("model.") :] if key.startswith("model.") else key
            inner[f"model.language_model.{bare}"] = value
        state = inner

    path = os.path.join(out_dir, name)
    os.makedirs(path, exist_ok=True)

    from safetensors.torch import save_file

    save_file(state, os.path.join(path, "model.safetensors"))

    cfg_out = dict(BASE_CONFIG)
    cfg_out["architectures"] = ["Qwen3_5ForCausalLM"]
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fp:
        json.dump(cfg_out, fp, indent=2)

    tokenizer = {
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
            "vocab": {f"t{i}": i for i in range(BASE_CONFIG["vocab_size"])},
            "merges": [],
        },
    }
    with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as fp:
        json.dump(tokenizer, fp)

    print(f"saved {path}: {len(state)} tensors, vlm_prefix={vlm_prefix}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/q35")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    build(args.out, "model-a", seed=0, vlm_prefix=False, with_mtp=True)
    build(args.out, "model-b", seed=1, vlm_prefix=False, with_mtp=True)
    build(args.out, "model-c-vlm", seed=2, vlm_prefix=True, with_mtp=True)

    moe_yaml = f"""\
base_model: {args.out}/model-a
experts:
  - source_model: {args.out}/model-a
    positive_prompts: ["write a python function", "debug this code"]
  - source_model: {args.out}/model-b
    positive_prompts: ["describe a sunset", "write a poem"]
gate_mode: random
experts_per_token: 2
dtype: float32
"""
    with open(os.path.join(args.out, "moe.yml"), "w", encoding="utf-8") as fp:
        fp.write(moe_yaml)
    print(f"wrote {os.path.join(args.out, 'moe.yml')}")

    moe_vlm_yaml = f"""\
base_model: {args.out}/model-c-vlm
experts:
  - source_model: {args.out}/model-c-vlm
    positive_prompts: ["describe a sunset", "write a poem"]
  - source_model: {args.out}/model-b
    positive_prompts: ["write a python function", "debug this code"]
gate_mode: random
experts_per_token: 2
dtype: float32
"""
    with open(os.path.join(args.out, "moe_vlm.yml"), "w", encoding="utf-8") as fp:
        fp.write(moe_vlm_yaml)
    print(f"wrote {os.path.join(args.out, 'moe_vlm.yml')}")
    print("\nNow run:")
    print(f"  mixtureforge-moe {args.out}/moe.yml {args.out}/out-moe --copy-tokenizer")


if __name__ == "__main__":
    main()
