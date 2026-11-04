# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for Qwen3.5 -> Qwen3.5-MoE conversion.

These cover the layout quirks that make Qwen3.5 different from Qwen3:

* hybrid ``layer_types`` (linear attention + every 4th full-attention layer),
* ``mtp.*`` checkpoint tensors that must survive the conversion,
* VLM checkpoints stored under ``model.language_model.*``.
"""

import json
import os
import tempfile

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from mixtureforge.moe.qwen3_5 import (
    Qwen3_5MoE,
    candidate_names,
    layer_kinds,
    strip_lm_prefix,
)

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
    "hidden_size": 128,
    "initializer_range": 0.02,
    "intermediate_size": 256,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 32,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 8,
    "linear_value_head_dim": 32,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 128,
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
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 64,
    "architectures": ["Qwen3_5ForCausalLM"],
}


# --------------------------------------------------------------------------- #
# name handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("model.embed_tokens.weight", "model.embed_tokens.weight"),
        (
            "model.layers.3.self_attn.q_proj.weight",
            "model.layers.3.self_attn.q_proj.weight",
        ),
        ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight"),
        (
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "model.layers.3.self_attn.q_proj.weight",
        ),
        (
            "language_model.model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ),
        ("model.language_model.lm_head.weight", "lm_head.weight"),
        ("lm_head.weight", "lm_head.weight"),
        ("mtp.fc.weight", "mtp.fc.weight"),
    ],
)
def test_strip_lm_prefix(raw, expected):
    assert strip_lm_prefix(raw) == expected


def test_candidate_names_roundtrip_vlm():
    """Every canonical name must be findable under its VLM spelling."""
    for canonical in (
        "model.embed_tokens.weight",
        "model.layers.3.self_attn.q_proj.weight",
        "lm_head.weight",
    ):
        vlm_names = [
            name
            for name in candidate_names(canonical)
            if name.startswith(("model.language_model.", "language_model."))
        ]
        assert vlm_names, canonical
        for vlm_name in vlm_names:
            assert strip_lm_prefix(vlm_name) == canonical


def test_layer_kinds():
    cfg = AutoConfig.for_model(
        "qwen3_5_text", **{k: v for k, v in BASE_CONFIG.items() if k != "model_type"}
    )
    assert layer_kinds(cfg, 4) == [
        "linear_attn",
        "linear_attn",
        "linear_attn",
        "self_attn",
    ]


def test_layer_kinds_defaults_to_full_attention_without_layer_types():
    class FakeCfg:
        pass

    assert layer_kinds(FakeCfg(), 3) == ["self_attn"] * 3


# --------------------------------------------------------------------------- #
# capability check
# --------------------------------------------------------------------------- #
def _write_tiny_model(path: str, model_type: str = "qwen3_5_text") -> str:
    cfg = dict(BASE_CONFIG)
    cfg["model_type"] = model_type
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fp:
        json.dump(cfg, fp)
    save_file(
        {"model.embed_tokens.weight": torch.zeros(4, 4)},
        os.path.join(path, "model.safetensors"),
    )
    return path


def test_supports_config_accepts_qwen3_5():
    from mixtureforge.common import ModelReference
    from mixtureforge.moe.config import Expert, MoEMergeConfig

    with tempfile.TemporaryDirectory() as tmp:
        model = _write_tiny_model(os.path.join(tmp, "m"))
        cfg = MoEMergeConfig(
            base_model=ModelReference.parse(model),
            experts=[
                Expert(
                    source_model=ModelReference.parse(model), positive_prompts=["a"]
                ),
                Expert(
                    source_model=ModelReference.parse(model), positive_prompts=["b"]
                ),
            ],
            gate_mode="random",
        )
        assert Qwen3_5MoE().supports_config(cfg) is True


def test_supports_config_rejects_other_architectures():
    from mixtureforge.common import ModelReference
    from mixtureforge.moe.config import Expert, MoEMergeConfig

    with tempfile.TemporaryDirectory() as tmp:
        model = _write_tiny_model(os.path.join(tmp, "m"), model_type="llama")
        cfg = MoEMergeConfig(
            base_model=ModelReference.parse(model),
            experts=[
                Expert(
                    source_model=ModelReference.parse(model), positive_prompts=["a"]
                ),
                Expert(
                    source_model=ModelReference.parse(model), positive_prompts=["b"]
                ),
            ],
            gate_mode="random",
        )
        assert Qwen3_5MoE().supports_config(cfg) is False


# --------------------------------------------------------------------------- #
# output config
# --------------------------------------------------------------------------- #
def test_generate_config_sets_moe_fields():
    from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig

    cfg = AutoConfig.for_model(
        "qwen3_5_text", **{k: v for k, v in BASE_CONFIG.items() if k != "model_type"}
    )
    out = Qwen3_5MoE()._generate_config(cfg, num_experts=4, experts_per_token=2)

    assert isinstance(out, Qwen3_5MoeTextConfig)
    assert out.model_type == "qwen3_5_moe_text"
    assert out.architectures == ["Qwen3_5MoeForCausalLM"]
    assert out.num_experts == 4
    assert out.num_experts_per_tok == 2
    assert out.moe_intermediate_size == cfg.intermediate_size
    assert out.shared_expert_intermediate_size == cfg.intermediate_size
    # hybrid layout survives
    assert out.layer_types == cfg.layer_types

    # and it round-trips through the strict config class
    round_tripped = Qwen3_5MoeTextConfig(**out.to_dict())
    assert round_tripped.num_experts == 4


def test_generate_config_clamps_experts_per_token():
    cfg = AutoConfig.for_model(
        "qwen3_5_text", **{k: v for k, v in BASE_CONFIG.items() if k != "model_type"}
    )
    out = Qwen3_5MoE()._generate_config(cfg, num_experts=2, experts_per_token=8)
    assert out.num_experts_per_tok == 2


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def _make_tiny_checkpoint(path: str, seed: int) -> str:
    """Real 4-layer hybrid Qwen3.5 checkpoint plus a fake MTP head."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "demo"))
    from make_tiny_qwen35 import build

    out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    return build(
        out_dir, os.path.basename(path), seed=seed, vlm_prefix=False, with_mtp=True
    )


def test_qwen3_5_moe_end_to_end():
    """Full conversion of a tiny hybrid checkpoint, then load + forward."""
    from transformers import AutoModelForCausalLM

    from mixtureforge.common import ModelReference
    from mixtureforge.moe.config import Expert, MoEMergeConfig
    from mixtureforge.options import MergeOptions
    from mixtureforge.scripts.moe import build as moe_build

    with tempfile.TemporaryDirectory() as tmp:
        a = _make_tiny_checkpoint(os.path.join(tmp, "a"), seed=0)
        b = _make_tiny_checkpoint(os.path.join(tmp, "b"), seed=1)
        out = os.path.join(tmp, "out")

        cfg = MoEMergeConfig(
            base_model=ModelReference.parse(a),
            experts=[
                Expert(source_model=ModelReference.parse(a), positive_prompts=["code"]),
                Expert(
                    source_model=ModelReference.parse(b), positive_prompts=["poetry"]
                ),
            ],
            gate_mode="random",
            experts_per_token=2,
            dtype="float32",
        )
        moe_build(
            cfg,
            out,
            MergeOptions(device="cpu", copy_tokenizer=True),
            load_in_4bit=False,
            load_in_8bit=False,
            device="cpu",
        )

        out_cfg = AutoConfig.from_pretrained(out)
        assert out_cfg.model_type == "qwen3_5_moe_text"
        assert out_cfg.num_experts == 2
        assert out_cfg.layer_types == BASE_CONFIG["layer_types"]

        tensors = load_file(os.path.join(out, "model.safetensors"))

        # hybrid attention: per-layer flavours, no cross-contamination
        assert "model.layers.0.linear_attn.in_proj_qkv.weight" in tensors
        assert "model.layers.3.self_attn.q_proj.weight" in tensors
        assert "model.layers.0.self_attn.q_proj.weight" not in tensors
        assert "model.layers.3.linear_attn.in_proj_qkv.weight" not in tensors

        # fused expert layout
        gu = tensors["model.layers.0.mlp.experts.gate_up_proj"]
        assert gu.shape[0] == 2 and gu.shape[1] == 2 * out_cfg.intermediate_size

        # shared expert gate has the shape the model expects
        gate = tensors["model.layers.0.mlp.shared_expert_gate.weight"]
        assert gate.shape == (1, out_cfg.hidden_size)

        # MTP head survived the conversion
        mtp_keys = [k for k in tensors if k.startswith("mtp.")]
        assert len(mtp_keys) == 15, mtp_keys

        # and the checkpoint actually loads and runs
        model = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([[5, 12, 99, 7]])).logits
        assert torch.isfinite(logits).all()


def test_qwen3_5_moe_accepts_vlm_prefixed_checkpoint():
    """The same conversion must work when tensors sit under model.language_model."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "demo"))
    from make_tiny_qwen35 import build

    from mixtureforge.common import ModelReference
    from mixtureforge.moe.config import Expert, MoEMergeConfig
    from mixtureforge.options import MergeOptions
    from mixtureforge.scripts.moe import build as moe_build

    with tempfile.TemporaryDirectory() as tmp:
        vlm = build(tmp, "vlm", seed=2, vlm_prefix=True, with_mtp=True)
        ref = ModelReference.parse(vlm)
        cfg = MoEMergeConfig(
            base_model=ref,
            experts=[
                Expert(source_model=ref, positive_prompts=["a"]),
                Expert(source_model=ref, positive_prompts=["b"]),
            ],
            gate_mode="random",
            experts_per_token=2,
            dtype="float32",
        )
        out = os.path.join(tmp, "out")
        moe_build(
            cfg,
            out,
            MergeOptions(device="cpu", copy_tokenizer=True),
            load_in_4bit=False,
            load_in_8bit=False,
            device="cpu",
        )

        tensors = load_file(os.path.join(out, "model.safetensors"))
        assert "model.embed_tokens.weight" in tensors
        assert "lm_head.weight" in tensors
        assert len([k for k in tensors if k.startswith("mtp.")]) == 15
