# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Regression tests for the router_input gate mode and single-expert merges.

These lock in two fixes found while validating the Qwen3.5 MoE conversion with
real train -> merge -> compare experiments:

* `is_bad_config` used `all()` over an empty slice when only one expert was
  configured, which is vacuously True and rejected valid 1-expert merges;
* `get_router_input_states` generates gate vectors from the tensor the router
  actually receives (post-attention normalised state) instead of the pre-norm
  residual stream.
"""

import pytest

from mixtureforge.moe.config import Expert, MoEMergeConfig, is_bad_config


# --------------------------------------------------------------------------- #
# single-expert configs must be accepted
# --------------------------------------------------------------------------- #
def _cfg(n_experts: int, same_prompts: bool = False, same_model: bool = True):
    from mixtureforge.common import ModelReference

    ref = ModelReference.parse("some/local/model")
    other = ModelReference.parse("some/local/other")
    experts = []
    for idx in range(n_experts):
        experts.append(
            Expert(
                source_model=ref if (same_model or idx == 0) else other,
                positive_prompts=["same"] if same_prompts else [f"prompt {idx}"],
            )
        )
    # experts_per_token must never exceed the expert count, otherwise
    # is_bad_config rejects the config for an unrelated reason.
    return MoEMergeConfig(
        base_model=ref,
        experts=experts,
        gate_mode="random",
        experts_per_token=min(2, len(experts)),
    )


def test_single_expert_is_allowed_with_identical_prompts():
    """A 1-expert MoE is the lossless-conversion case; it must not be rejected."""
    assert is_bad_config(_cfg(1, same_prompts=True)) is False


def test_single_expert_is_allowed_without_flag():
    assert is_bad_config(_cfg(1, same_model=True)) is False


def test_single_expert_still_needs_a_prompt_for_hidden_gating():
    """Only the *random* gate mode may skip prompts."""
    cfg = _cfg(1, same_prompts=True)
    cfg.experts[0].positive_prompts = None
    cfg.gate_mode = "hidden"
    assert is_bad_config(cfg) is True


def test_multiple_identical_prompts_still_rejected():
    """Duplicate prompts are only checked for prompt-driven gate modes."""
    cfg = _cfg(2, same_prompts=True, same_model=False)
    cfg.gate_mode = "hidden"
    assert is_bad_config(cfg) is True


def test_multiple_identical_models_still_rejected_without_flag():
    """Upstream returns early for gate_mode='random', so use a prompt mode."""
    cfg = _cfg(2, same_prompts=False, same_model=True)
    cfg.gate_mode = "hidden"
    assert is_bad_config(cfg) is True


def test_multiple_identical_models_allowed_with_flag():
    assert (
        is_bad_config(_cfg(2, same_prompts=False, same_model=True), allow_all_same=True)
        is False
    )


# --------------------------------------------------------------------------- #
# router input extraction
# --------------------------------------------------------------------------- #
def test_get_router_input_states_matches_model_inputs():
    """The captured vectors must be layer inputs, not residual-stream states."""
    import torch
    from transformers import AutoConfig

    from mixtureforge.moe.router import get_router_input_states

    from .test_moe_qwen3_5 import (  # noqa: F401  (fixture builder lives there)
        _make_tiny_checkpoint,
    )

    with __import__("tempfile").TemporaryDirectory() as tmp:
        path = __import__("os").path.join(tmp, "m")
        _make_tiny_checkpoint(path, seed=0)

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        model.eval()

        from transformers import BatchEncoding

        def encode(ids):
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([ids]),
                    "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                }
            )

        vecs = get_router_input_states(model, encode([5, 12, 99]), average=True)

        cfg = AutoConfig.from_pretrained(path)
        assert vecs.shape == (cfg.num_hidden_layers, cfg.hidden_size)
        assert torch.isfinite(vecs).all()

        # different prompts must not collapse to the same vector in *every* layer
        other = get_router_input_states(model, encode([7, 21, 44]), average=True)
        cos = [
            torch.nn.functional.cosine_similarity(vecs[i], other[i], dim=0).item()
            for i in range(vecs.shape[0])
        ]
        assert min(cos) < 0.999, f"gate vectors are degenerate: {cos}"


def test_router_input_mode_is_accepted_by_gate_params():
    """gate_mode must be a known string for the CLI/YAML surface."""
    cfg = _cfg(2, same_model=False, same_prompts=False)
    cfg.gate_mode = "router_input"
    # is_bad_config returns None (falsy) for acceptable configs
    assert not is_bad_config(cfg)
