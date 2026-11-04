# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Pre-flight check for a 27B class `mergekit-moe` run.

A real Qwen3.5-27B MoE merge is expensive: every expert is ~49 GB of download
and the output is ~195 GB. This script answers "will it work, and will it fit?"
*before* any weight is downloaded, using only config files (a few KB each).

It checks, in order:

1. every referenced model resolves and its `model_type` is a legal dense input
2. the selected output architecture accepts the config (same logic as
   `mergekit-moe`, so a rejection here is a rejection there)
3. `vocab_size`, `hidden_size`, `num_hidden_layers` and `layer_types` agree
   across base and experts -- mismatches produce a silently broken model
4. the merged MoE geometry and the resulting parameter count
5. free disk and RAM against what the merge actually needs

    python tools/demo/preflight_moe.py moe.yml
    python tools/demo/preflight_moe.py moe.yml --out-dir /data/qwen38-moe

Exit code is 1 when the run would fail or not fit.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Dict, List, Optional

import yaml

from mergekit.common import ModelReference
from mergekit.moe.config import Expert, MoEMergeConfig
from mergekit.moe.qwen3_5 import SUPPORTED_MODEL_TYPES, Qwen3_5MoE
from mergekit.options import MergeOptions

# architecture -> the dense model types it can turn into experts
DENSE_INPUTS: Dict[str, tuple] = {
    "Mixtral": ("llama", "mistral"),
    "DeepSeek MoE": ("llama", "mistral"),
    "Qwen MoE": ("llama", "mistral", "qwen2"),
    "Qwen3 MoE": ("qwen3",),
    "Qwen3.5 MoE": SUPPORTED_MODEL_TYPES,
}

OK = "  [ok]  "
BAD = "  [FAIL]"
WARN = "  [warn]"


def _get_cfg(ref: ModelReference, trust_remote_code: bool = False):
    return ref.config(trust_remote_code=trust_remote_code)


def _human(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def run(config_path: str, out_dir: Optional[str], verbose: bool = True) -> int:
    with open(config_path, "r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    config = MoEMergeConfig.model_validate(raw)

    problems: List[str] = []
    warnings: List[str] = []

    def say(tag: str, msg: str) -> None:
        if verbose:
            print(f"{tag} {msg}")

    # ---------------------------------------------------------------- 1. inputs
    print("\n[1] input models")
    refs: List[ModelReference] = [config.base_model]
    refs += [e.source_model for e in config.experts]
    refs += [e.source_model for e in (config.shared_experts or [])]

    seen = {}
    cfgs = {}
    for ref in refs:
        key = str(ref)
        if key in seen:
            cfgs[key] = seen[key]
            continue
        try:
            cfg = _get_cfg(ref)
        except Exception as exc:
            problems.append(f"cannot read config for {key}: {exc}")
            say(BAD, f"{key}: unreadable ({type(exc).__name__})")
            continue
        seen[key] = cfg
        cfgs[key] = cfg
        say(OK, f"{key}: model_type={cfg.model_type}")

    if problems:
        print("\nCannot continue without readable configs.")
        return 1

    # ------------------------------------------------- 2. output architecture
    print("\n[2] output architecture")
    arch = Qwen3_5MoE()
    if arch.supports_config(config, explain=True):
        say(OK, f"{arch.name()} accepts this config")
    else:
        problems.append(
            "the output architecture rejects this config (see warnings above)"
        )
        say(BAD, f"{arch.name()} rejects this config")

    types = {cfg.model_type for cfg in cfgs.values()}
    allowed = set()
    for t in DENSE_INPUTS.get(arch.name(), ()):
        allowed.add(t)
    if types & {"qwen3_5_moe_text", "qwen3_5_moe"}:
        problems.append(
            "a Qwen3.5-MoE checkpoint is in the inputs; only dense Qwen3.5 "
            "models can become experts (there is no dense MLP to lift)"
        )
    if not types <= allowed:
        warnings.append(f"unexpected model types: {sorted(types - allowed)}")

    # ------------------------------------------------------ 3. shape agreement
    print("\n[3] shape agreement across inputs")
    for field in ("vocab_size", "hidden_size", "num_hidden_layers"):
        values = {str(k): getattr(c, field, None) for k, c in cfgs.items()}
        distinct = {v for v in values.values() if v is not None}
        if len(distinct) > 1:
            problems.append(f"{field} differs across models: {values}")
            say(BAD, f"{field}: {values}")
        elif distinct:
            say(OK, f"{field} = {distinct.pop()} everywhere")

    layer_types = {}
    for key, cfg in cfgs.items():
        lt = getattr(cfg, "layer_types", None)
        layer_types[key] = tuple(lt) if lt else None
    distinct_lt = {v for v in layer_types.values() if v is not None}
    if len(distinct_lt) > 1:
        problems.append(f"layer_types differ across models: {layer_types}")
        say(BAD, "layer_types differ (hybrid pattern must match exactly)")
    elif distinct_lt:
        lt = distinct_lt.pop()
        linear = sum(1 for x in lt if "linear" in x)
        say(OK, f"layer_types identical: {linear} linear + {len(lt) - linear} full")

    base_cfg = cfgs[str(config.base_model)]
    inter = getattr(base_cfg, "intermediate_size", None)
    hidden = getattr(base_cfg, "hidden_size", 0)
    layers = getattr(base_cfg, "num_hidden_layers", 0)
    vocab = getattr(base_cfg, "vocab_size", 0)

    # --------------------------------------------------------- 4. output size
    print("\n[4] merged geometry")
    n_exp = len(config.experts)
    if inter:
        # dense model = embeddings + attention/linear + ONE mlp per layer.
        # the MoE output swaps that single mlp for n_exp + shared + router.
        base_params = _estimate_dense_params(base_cfg, inter, hidden, layers, vocab)
        dense_mlps = layers * 3 * hidden * inter
        moe_mlps = (
            layers * (n_exp + (1 if config.shared_experts else 0)) * 3 * hidden * inter
        )
        moe_mlps += layers * (n_exp + 1) * hidden  # routers
        moe_params = base_params - dense_mlps + moe_mlps
        say(
            OK,
            f"experts={n_exp} topk={config.experts_per_token} "
            f"gate_mode={config.gate_mode}",
        )
        say(
            OK,
            f"dense {base_params / 1e9:.2f}B -> MoE {moe_params / 1e9:.2f}B "
            f"({moe_params * 2 / 1e9:.0f} GB bf16)",
        )
        if n_exp & (n_exp - 1):
            warnings.append(
                f"{n_exp} experts is not a power of two; llama.cpp cannot load this"
            )
        if config.experts_per_token > n_exp:
            problems.append(
                f"experts_per_token={config.experts_per_token} > experts={n_exp}"
            )

    # ------------------------------------------------------- 5. disk and RAM
    print("\n[5] disk and memory")
    dense_gb = _estimate_size(cfgs[str(config.base_model)])
    unique_inputs = len({str(r) for r in refs})
    out_gb = dense_gb * (n_exp + (1 if config.shared_experts else 0))
    need_in = dense_gb * unique_inputs
    need_out = out_gb
    total = need_in + need_out

    target = out_dir or os.path.abspath("./merged-moe")
    probe = target
    while not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    free_gb = shutil.disk_usage(probe).free / 1e9

    say(OK, f"input weights  ~{need_in:.0f} GB ({unique_inputs} unique models)")
    say(OK, f"output weights ~{need_out:.0f} GB")
    if free_gb < total:
        problems.append(
            f"not enough disk: need ~{total:.0f} GB, {free_gb:.0f} GB free at {probe}"
        )
        say(BAD, f"need ~{total:.0f} GB, have {free_gb:.0f} GB")
    else:
        say(OK, f"need ~{total:.0f} GB, {free_gb:.0f} GB free")

    if config.gate_mode in ("hidden", "hidden_avg", "hidden_last", "router_input"):
        ram_gb = dense_gb
        have = _available_ram_gb()
        if have is not None:
            if have < ram_gb:
                problems.append(
                    f"gate_mode '{config.gate_mode}' loads a whole dense model "
                    f"(~{ram_gb:.0f} GB); only {have:.0f} GB RAM available"
                )
                say(
                    BAD,
                    f"gate extraction needs ~{ram_gb:.0f} GB RAM, have {have:.0f} GB",
                )
            else:
                say(OK, f"gate extraction ~{ram_gb:.0f} GB RAM, have {have:.0f} GB")
    else:
        say(OK, f"gate_mode '{config.gate_mode}' needs no model load")

    # ------------------------------------------------------------- verdict
    print("\n" + "=" * 66)
    for w in warnings:
        print(f"{WARN} {w}")
    if problems:
        print(f"\n{len(problems)} problem(s) — this run would fail or not fit:")
        for p in problems:
            print(f"   - {p}")
        print("=" * 66)
        return 1
    print("READY — no blocking problems found.")
    print("=" * 66)
    return 0


def _estimate_dense_params(cfg, inter, hidden, layers, vocab) -> float:
    """Rough dense parameter count, good enough for a disk/RAM budget."""
    embeddings = (
        vocab * hidden * (1 if getattr(cfg, "tie_word_embeddings", False) else 2)
    )
    heads = getattr(cfg, "num_attention_heads", hidden // getattr(cfg, "head_dim", 64))
    kv = getattr(cfg, "num_key_value_heads", heads)
    head_dim = getattr(cfg, "head_dim", hidden // max(heads, 1))
    q_mult = 2 if getattr(cfg, "attn_output_gate", False) else 1

    full_attn = q_mult * heads * head_dim * hidden + 2 * kv * head_dim * hidden
    full_attn += hidden * heads * head_dim + 2 * head_dim

    k_heads = getattr(cfg, "linear_num_key_heads", 0)
    k_dim = getattr(cfg, "linear_key_head_dim", 0)
    v_heads = getattr(cfg, "linear_num_value_heads", 0)
    v_dim = getattr(cfg, "linear_value_head_dim", 0)
    conv = getattr(cfg, "linear_conv_kernel_dim", 0)
    linear = (
        hidden * k_heads * k_dim
        + hidden * v_heads * v_dim
        + hidden * v_heads * v_dim
        + v_heads * v_dim * hidden // max(heads, 1)
    )
    linear += (
        k_heads * k_dim + 2 * v_heads + v_heads * v_dim * conv + 2 * v_heads * hidden
    )
    linear += v_heads * v_dim  # per-head gated norm

    types = getattr(cfg, "layer_types", None) or []
    n_linear = sum(1 for t in types if "linear" in t) if types else layers // 2
    n_full = layers - n_linear if types else layers // 2

    mlp = 3 * hidden * inter
    norms = 2 * hidden
    attn_total = n_linear * (linear + norms) + n_full * (full_attn + norms)
    return embeddings + attn_total + layers * mlp + hidden


def _estimate_size(cfg) -> float:
    hidden = getattr(cfg, "hidden_size", 4096)
    inter = getattr(cfg, "intermediate_size", 4 * hidden)
    layers = getattr(cfg, "num_hidden_layers", 32)
    vocab = getattr(cfg, "vocab_size", 32000)
    params = _estimate_dense_params(cfg, inter, hidden, layers, vocab)
    return params * 2 / 1e9  # bf16


def _available_ram_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fp:
            for line in fp:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024 / 1024
    except OSError:
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to the moe yaml")
    parser.add_argument("--out-dir", default=None, help="where the merge will go")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    sys.exit(run(args.config, args.out_dir, verbose=not args.quiet))


if __name__ == "__main__":
    main()
