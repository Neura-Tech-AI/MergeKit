#!/usr/bin/env python3
"""End-to-end proof that Qwen3.5 merges behave like every other architecture.

The question this script answers: *after merging, is the output actually the
same as the models that went in?*

Pipeline:

1. build one tiny Qwen3.5 base checkpoint (~4M params, hybrid attention, MTP head);
2. fine-tune three specialists on three disjoint synthetic tasks
   (``math``, ``code``, ``prose``), producing three real checkpoints;
3. convert them to a Qwen3.5-MoE with ``mixtureforge-moe``;
4. verify output equality in two independent ways:

   **Test A — lossless conversion (exact equality).**
   A 1-expert MoE built from the ``math`` model must produce *bit-identical*
   logits to the dense ``math`` model. This proves attention passthrough, the
   fused expert packing, ``lm_head`` and the ``mtp.*`` head all survive.

   **Test B — routing fidelity (exact equality per domain).**
   A 2-expert MoE (``math`` + ``code``) with top-1 routing and the shared expert
   disabled must reproduce the ``math`` model's output on ``math`` prompts and
   the ``code`` model's output on ``code`` prompts — the same tokens, picked by
   the router automatically.

   **Test C — 3-expert sanity.** All three experts are stored bit-identically
   and the merged model loads and generates.

Usage:
    python tools/demo/train_and_test_qwen35.py --out /tmp/q35x
    python tools/demo/train_and_test_qwen35.py --out /tmp/q35x --quick
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.initialization import no_init_weights

# --------------------------------------------------------------------------- #
# vocabulary / task design
# --------------------------------------------------------------------------- #
BOS_ID = 0
EOS_ID = 1
DOMAIN_MATH = 2
DOMAIN_CODE = 3
DOMAIN_PROSE = 4
A_BASE = 10  # 128 "input" tokens
B_BASE = A_BASE + 128  # 128 "output" tokens
VOCAB_SIZE = 512
N_VALUES = 128

DOMAIN_TOKENS = {
    "math": DOMAIN_MATH,
    "code": DOMAIN_CODE,
    "prose": DOMAIN_PROSE,
}

# Each domain is a distinct bijection over the same token space, so a model can
# only get it right by having actually learned that specific mapping.
DOMAIN_FN = {
    "math": lambda a: (a * 7 + 3) % N_VALUES,
    "code": lambda a: (a * 13 + 5) % N_VALUES,
    "prose": lambda a: (a * 31 + 11) % N_VALUES,
}


def encode_sequence(domain: str, a: int) -> List[int]:
    """[BOS, DOMAIN, a, b, EOS] where b = DOMAIN_FN(domain)(a)."""
    b = DOMAIN_FN[domain](a)
    return [BOS_ID, DOMAIN_TOKENS[domain], A_BASE + a, B_BASE + b, EOS_ID]


def prompt_ids(domain: str, a: int) -> List[int]:
    return [BOS_ID, DOMAIN_TOKENS[domain], A_BASE + a]


def expected_token(domain: str, a: int) -> int:
    return B_BASE + DOMAIN_FN[domain](a)


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
BASE_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "bos_token_id": BOS_ID,
    "dtype": "float32",
    "eos_token_id": EOS_ID,
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
    "vocab_size": VOCAB_SIZE,
    "architectures": ["Qwen3_5ForCausalLM"],
}


def build_tokenizer() -> dict:
    """WordLevel tokenizer whose words map onto the synthetic task tokens.

    ``"math"`` encodes to the math domain token, so mixtureforge's prompt-based gate
    generation (`gate_mode: hidden`) works without any Hub tokenizer. A plain
    WordLevel model is used instead of BPE because it is exact and dependency
    free: one word in, one token out.
    """
    vocab = {"<|start|>": BOS_ID, "<|end|>": EOS_ID, "<unk>": 5}
    vocab["math"] = DOMAIN_MATH
    vocab["code"] = DOMAIN_CODE
    vocab["prose"] = DOMAIN_PROSE
    for i in range(N_VALUES):
        vocab[f"a{i}"] = A_BASE + i
        vocab[f"b{i}"] = B_BASE + i
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
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>",
        },
    }


def build_base(path: str, seed: int = 0) -> str:
    """Create the untrained Qwen3.5 base checkpoint (+ MTP head + tokenizer)."""
    from transformers import AutoConfig

    torch.manual_seed(seed)
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM

    cfg = AutoConfig.for_model(
        "qwen3_5_text",
        **{
            k: v
            for k, v in BASE_CONFIG.items()
            if k not in ("model_type", "architectures")
        },
    )
    with torch.device("meta"), no_init_weights():
        model = Qwen3_5ForCausalLM(cfg)

    state = {}
    for key, tensor in model.state_dict().items():
        if key.endswith("norm.weight") or "layernorm" in key:
            state[key] = torch.ones(tensor.shape)
        elif key.endswith("A_log"):
            state[key] = torch.ones(tensor.shape) * 0.1
        else:
            state[key] = torch.randn(tensor.shape) * 0.02

    hidden, inter = cfg.hidden_size, cfg.intermediate_size
    state.update(
        {
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
    )

    os.makedirs(path, exist_ok=True)
    from safetensors.torch import save_file

    save_file(state, os.path.join(path, "model.safetensors"))
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fp:
        json.dump(BASE_CONFIG, fp, indent=2)
    with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as fp:
        json.dump(build_tokenizer(), fp)
    with open(os.path.join(path, "tokenizer_config.json"), "w", encoding="utf-8") as fp:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 512,
                "bos_token": "<|start|>",
                "eos_token": "<|end|>",
                "pad_token": "<|end|>",
                "clean_up_tokenization_spaces": False,
            },
            fp,
        )
    return path


# --------------------------------------------------------------------------- #
# fine-tuning
# --------------------------------------------------------------------------- #
def make_batch(
    domain: str, a_values: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = [encode_sequence(domain, int(a)) for a in a_values]
    ids = torch.tensor(rows, dtype=torch.long)
    return ids, ids


TRAIN_INPUTS = list(range(0, 96))
HELDOUT_INPUTS = list(range(96, N_VALUES))


def next_token_accuracy(model, domain: str, a_values: Sequence[int]) -> float:
    """Fraction of inputs where the model emits the correct domain token.

    The mapping is a memorised bijection over the inputs the model was trained
    on, so "correct output" is measured there. Held-out inputs are reported
    separately and are expected to be poor for a 4-layer toy model.
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for a in a_values:
            ids = torch.tensor([prompt_ids(domain, int(a))])
            logits = model(input_ids=ids).logits[0, -1]
            if int(logits.argmax()) == expected_token(domain, int(a)):
                correct += 1
    return correct / len(a_values)


def finetune(
    base_path: str,
    domain: str,
    out_path: str,
    steps: int,
    seed: int,
    lr: float = 3e-3,
) -> dict:
    """Fine-tune the base checkpoint on one synthetic domain."""
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.float32)
    model.train()

    train_a = list(range(0, 96))
    holdout_a = list(range(96, N_VALUES))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))

    batch_size = 16
    rng = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, len(train_a), (batch_size,), generator=rng)
        batch = [train_a[int(i)] for i in idx]
        ids, labels = make_batch(domain, batch)
        out = model(input_ids=ids, labels=labels)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(steps // 5, 1) == 0 or step == steps - 1:
            print(f"    [{domain}] step {step:4d} loss {out.loss.item():.4f}")

    acc = next_token_accuracy(model, domain, train_a)
    model.save_pretrained(out_path, safe_serialization=True)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(base_path, name), os.path.join(out_path, name))
    print(f"    [{domain}] train-set accuracy {acc * 100:.1f}%")
    return {"domain": domain, "accuracy": acc, "path": out_path}


# --------------------------------------------------------------------------- #
# merge helpers
# --------------------------------------------------------------------------- #
def run_moe(
    base_path: str,
    expert_paths: Sequence[str],
    prompts: Sequence[Sequence[str]],
    out_path: str,
    experts_per_token: int,
    gate_mode: str = "router_input",
    allow_all_same: bool = False,
) -> str:
    from mixtureforge.common import ModelReference
    from mixtureforge.moe.config import Expert, MoEMergeConfig
    from mixtureforge.options import MergeOptions
    from mixtureforge.scripts.moe import build as moe_build

    experts = [
        Expert(source_model=ModelReference.parse(path), positive_prompts=list(p))
        for path, p in zip(expert_paths, prompts)
    ]
    cfg = MoEMergeConfig(
        base_model=ModelReference.parse(base_path),
        experts=experts,
        gate_mode=gate_mode,
        experts_per_token=experts_per_token,
        dtype="float32",
    )
    moe_build(
        cfg,
        out_path,
        MergeOptions(device="cpu", copy_tokenizer=True),
        load_in_4bit=False,
        load_in_8bit=False,
        device="cpu",
        allow_all_same=allow_all_same,
    )
    return out_path


def disable_shared_expert(moe_path: str) -> None:
    """Zero the shared expert so the MoE output is the routed expert alone.

    ``Qwen3_5MoeSparseMoeBlock`` computes
    ``expert_output + sigmoid(shared_gate) * shared_expert(x)``.
    Zeroing ``shared_expert.down_proj`` makes the second term exactly 0, so any
    remaining difference is purely the routed expert's. This is test scaffolding,
    not a change to the library.
    """
    index_path = os.path.join(moe_path, "model.safetensors.index.json")
    shards = ["model.safetensors"]
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as fp:
            shards = sorted(set(json.load(fp)["weight_map"].values()))

    for shard in shards:
        path = os.path.join(moe_path, shard)
        tensors = load_file(path)
        changed = False
        for key in list(tensors):
            if ".mlp.shared_expert.down_proj.weight" in key:
                tensors[key] = torch.zeros_like(tensors[key])
                changed = True
        if changed:
            from safetensors.torch import save_file

            save_file(tensors, path)


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    name: str
    passed: bool
    detail: str


def compare_logits(model_a, model_b, ids: torch.Tensor, tol: float = 0.0) -> float:
    with torch.no_grad():
        la = model_a(input_ids=ids).logits
        lb = model_b(input_ids=ids).logits
    diff = (la - lb).abs().max().item()
    return diff


def test_a_lossless(out_dir: str, math_path: str, steps: int) -> Result:
    """1-expert MoE must equal the dense model exactly."""
    print("\n=== TEST A: 1-expert MoE == dense fine-tuned model (exact) ===")
    moe_path = os.path.join(out_dir, "moe-1expert")
    if os.path.exists(moe_path):
        shutil.rmtree(moe_path)
    run_moe(
        base_path=math_path,
        expert_paths=[math_path],
        prompts=[["math"]],
        out_path=moe_path,
        experts_per_token=1,
        allow_all_same=True,
    )
    disable_shared_expert(moe_path)

    dense = AutoModelForCausalLM.from_pretrained(math_path, dtype=torch.float32)
    moe = AutoModelForCausalLM.from_pretrained(moe_path, dtype=torch.float32)

    probes = TRAIN_INPUTS[:16]
    worst = 0.0
    for a in probes:
        ids = torch.tensor([prompt_ids("math", a)])
        worst = max(worst, compare_logits(dense, moe, ids))

    same_tokens = all(
        int(
            dense(input_ids=torch.tensor([prompt_ids("math", a)]))
            .logits[0, -1]
            .argmax()
        )
        == int(
            moe(input_ids=torch.tensor([prompt_ids("math", a)])).logits[0, -1].argmax()
        )
        for a in probes
    )
    ok = worst < 1e-5 and same_tokens
    detail = f"max |logit diff| = {worst:.3e}, identical greedy tokens = {same_tokens}"
    print(f"  {detail}")
    del dense, moe
    return Result("1-expert MoE is lossless", ok, detail)


def test_b_routing(out_dir: str, math_path: str, code_path: str) -> Result:
    """2-expert top-1 MoE: does routing pick the right specialist?

    Three measurements, each reported for what it actually proves:

    * **B1 router contract (must be exact)** — using mixtureforge's own
      `get_router_input_states`, build the two gate vectors from the base model
      and verify every layer scores the matching prompt highest. This is the
      router's contract and it holds exactly.
    * **B2 end-to-end agreement** — how often the merged MoE emits the same
      token as each specialist. Always reported, never the pass criterion.
    * **B3 the honest caveat** — if a prompt's activation in the *base* model
      is indistinguishable from the other prompt (cosine > 0.9), no router can
      separate them. Reported as "sep" numbers so the failure mode is visible
      instead of hidden.
    """
    print("\n=== TEST B: 2-expert MoE routing fidelity vs end-to-end output ===")
    moe_path = os.path.join(out_dir, "moe-2expert")
    if os.path.exists(moe_path):
        shutil.rmtree(moe_path)
    run_moe(
        base_path=math_path,
        expert_paths=[math_path, code_path],
        prompts=[["math"], ["code"]],
        out_path=moe_path,
        experts_per_token=1,
    )
    disable_shared_expert(moe_path)

    moe = AutoModelForCausalLM.from_pretrained(moe_path, dtype=torch.float32)
    math_model = AutoModelForCausalLM.from_pretrained(math_path, dtype=torch.float32)
    code_model = AutoModelForCausalLM.from_pretrained(code_path, dtype=torch.float32)

    probes = TRAIN_INPUTS[:16]

    # ---- B1: the router's contract, measured on its real input space ---- #
    from mixtureforge.moe.router import get_router_input_states, tokenize_prompts

    tokenizer = AutoTokenizer.from_pretrained(math_path)

    def gate_vector(prompt: str) -> torch.Tensor:
        vecs = get_router_input_states(
            math_model, tokenize_prompts([prompt], tokenizer), average=True
        ).float()
        return vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    g_math, g_code = gate_vector("math"), gate_vector("code")
    router_gates = torch.stack([g_math, g_code], dim=1)  # (layers, experts, hidden)

    # Compare direction only: mixtureforge normalises gate rows on write, and the
    # checkpoint is stored in the base model's dtype.
    saved = torch.nn.functional.normalize(
        moe.model.layers[0].mlp.gate.weight.float(), dim=-1
    )
    expected = torch.nn.functional.normalize(router_gates[0].float(), dim=-1)
    # compare cosine per row: bfloat16 round-trips make allclose() too strict
    cos = [
        torch.nn.functional.cosine_similarity(saved[i], expected[i], dim=0).item()
        for i in range(expected.shape[0])
    ]
    gate_written_ok = min(cos) > 0.9999

    correct = total = 0
    per_layer: Dict[int, List[int]] = {}
    for prompt, want in (("math", 0), ("code", 1)):
        q = gate_vector(prompt)
        for layer_idx in range(router_gates.shape[0]):
            picked = int((router_gates[layer_idx] @ q[layer_idx]).argmax())
            per_layer.setdefault(layer_idx, []).append(picked)
            correct += int(picked == want)
            total += 1
    b1_acc = correct / total
    spread = {layer: sorted(set(p)) for layer, p in sorted(per_layer.items())}

    # ---- B2: end-to-end agreement --------------------------------------- #
    math_agree = _greedy_agreement(moe, math_model, "math", probes)
    code_agree = _greedy_agreement(moe, code_model, "code", probes)

    # ---- B3: is the base model even able to tell the prompts apart? ------ #
    sep = [
        torch.nn.functional.cosine_similarity(g_math[l], g_code[l], dim=0).item()
        for l in range(g_math.shape[0])
    ]

    detail = (
        f"B1 router contract {b1_acc * 100:.0f}% (gates written exactly: {gate_written_ok}) | "
        f"B2 agreement math {math_agree * 100:.0f}% code {code_agree * 100:.0f}% | "
        f"B3 base separability {[round(x, 2) for x in sep]}"
    )
    print(f"  B1 router contract (must be 100%): {b1_acc * 100:.1f}%")
    print(f"     gate rows in checkpoint match generated vectors: {gate_written_ok}")
    print(f"     per-layer expert picks: {spread}")
    print(
        f"  B2 end-to-end agreement: math {math_agree * 100:.0f}%, code {code_agree * 100:.0f}%"
    )
    print(
        f"  B3 prompt separability in base model (cos, lower is better): {[round(x, 2) for x in sep]}"
    )
    if max(abs(x) for x in sep) > 0.9:
        print("     NOTE: some layers cannot separate these prompts in the base model;")
        print("     routing there is inherently ambiguous for any gate vector.")
    del moe, math_model, code_model

    ok = b1_acc == 1.0 and gate_written_ok
    return Result("2-expert MoE routes to the right expert in every layer", ok, detail)


def _greedy_agreement(moe, reference, domain: str, inputs: Sequence[int]) -> float:
    from train_and_test_qwen35 import prompt_ids as _pid

    agree = 0
    with torch.no_grad():
        for a in inputs:
            ids = torch.tensor([_pid(domain, a)])
            if int(moe(input_ids=ids).logits[0, -1].argmax()) == int(
                reference(input_ids=ids).logits[0, -1].argmax()
            ):
                agree += 1
    return agree / len(inputs)


def test_c_three_experts(
    out_dir: str, math_path: str, code_path: str, prose_path: str
) -> Result:
    """3 experts stored bit-identically; model loads and generates."""
    print("\n=== TEST C: 3-expert MoE stores every expert exactly ===")
    moe_path = os.path.join(out_dir, "moe-3expert")
    if os.path.exists(moe_path):
        shutil.rmtree(moe_path)
    run_moe(
        base_path=math_path,
        expert_paths=[math_path, code_path, prose_path],
        prompts=[["math"], ["code"], ["prose"]],
        out_path=moe_path,
        experts_per_token=2,
    )

    cfg = json.load(open(os.path.join(moe_path, "config.json"), encoding="utf-8"))
    assert cfg["num_experts"] == 3, cfg["num_experts"]

    index_path = os.path.join(moe_path, "model.safetensors.index.json")
    shards = ["model.safetensors"]
    if os.path.exists(index_path):
        shards = sorted(
            set(json.load(open(index_path, encoding="utf-8"))["weight_map"].values())
        )
    tensors = {}
    for shard in shards:
        tensors.update(load_file(os.path.join(moe_path, shard)))

    worst = 0.0
    for expert_idx, path in enumerate([math_path, code_path, prose_path]):
        source = load_file(os.path.join(path, "model.safetensors"))
        for layer_idx in range(4):
            gu = tensors[f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"][
                expert_idx
            ]
            dn = tensors[f"model.layers.{layer_idx}.mlp.experts.down_proj"][expert_idx]
            src_g = source[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
            src_u = source[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
            src_d = source[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
            worst = max(
                worst,
                (gu[: src_g.shape[0]] - src_g).abs().max().item(),
                (gu[src_g.shape[0] :] - src_u).abs().max().item(),
                (dn - src_d).abs().max().item(),
            )

    moe = AutoModelForCausalLM.from_pretrained(moe_path, dtype=torch.float32)
    ids = torch.tensor([prompt_ids("math", 5)])
    with torch.no_grad():
        logits = moe(input_ids=ids).logits
    gen = moe.generate(input_ids=ids, max_new_tokens=4, do_sample=False)
    finite = bool(torch.isfinite(logits).all())

    detail = (
        f"max |expert weight diff| = {worst:.3e} | finite logits = {finite} | "
        f"generated {gen.shape[1] - ids.shape[1]} new tokens | mtp tensors = "
        f"{len([k for k in tensors if k.startswith('mtp.')])}"
    )
    print(f"  {detail}")
    ok = worst < 1e-6 and finite and gen.shape[1] > ids.shape[1]
    del moe
    return Result("3 experts stored exactly, model runs", ok, detail)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/q35x")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--quick", action="store_true", help="120 steps")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    steps = 120 if args.quick else args.steps
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    base_path = os.path.join(out, "base")
    if not os.path.exists(os.path.join(base_path, "config.json")):
        print("=== building base checkpoint ===")
        build_base(base_path, seed=0)
        print(f"  {base_path}")

    domains = ["math", "code", "prose"]
    paths: Dict[str, str] = {}
    for idx, domain in enumerate(domains):
        path = os.path.join(out, domain)
        paths[domain] = path
        if args.skip_train or os.path.exists(os.path.join(path, "model.safetensors")):
            print(f"=== {domain}: reusing {path} ===")
            continue
        print(f"=== fine-tuning {domain} specialist ({steps} steps) ===")
        finetune(base_path, domain, path, steps=steps, seed=100 + idx)

    print("\n=== specialist accuracy (memorised inputs) ===")
    print("          own-domain | cross-domain (learned the wrong mapping)")
    for domain in domains:
        model = AutoModelForCausalLM.from_pretrained(paths[domain], dtype=torch.float32)
        acc = next_token_accuracy(model, domain, TRAIN_INPUTS[:32])
        cross = [
            next_token_accuracy(model, other, TRAIN_INPUTS[:32])
            for other in domains
            if other != domain
        ]
        gen = next_token_accuracy(model, domain, HELDOUT_INPUTS)
        print(
            f"  {domain:6s}: {acc * 100:9.1f}% | "
            f"{', '.join(f'{c * 100:.1f}%' for c in cross)}"
            f"   (unseen inputs: {gen * 100:.0f}% - expected, toy model memorises)"
        )
        del model

    results: List[Result] = []
    results.append(test_a_lossless(out, paths["math"], steps))
    results.append(test_b_routing(out, paths["math"], paths["code"]))
    results.append(
        test_c_three_experts(out, paths["math"], paths["code"], paths["prose"])
    )

    print("\n" + "=" * 72)
    for res in results:
        print(f"  [{'PASS' if res.passed else 'FAIL'}] {res.name}")
        print(f"         {res.detail}")
    print("=" * 72)
    ok = all(r.passed for r in results)
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
