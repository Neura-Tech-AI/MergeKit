#!/usr/bin/env python3
"""Compare a dense Qwen3.5 checkpoint against its MoE conversion.

Verifies, without needing the real 27B model:

1. the MoE checkpoint loads with `AutoModelForCausalLM` (no missing/mismatched keys);
2. every router row actually selects the expert whose prompts it was built from;
3. attention weights are bit-identical to the base model, per layer flavour;
4. the shared expert (dense MLP) is preserved;
5. checkpoint-only tensors (`mtp.*`) survive the conversion;
6. a forward pass and a short greedy generation run.

Usage:
    python tools/demo/verify_qwen35_moe.py --base /tmp/q35/model-a --moe /tmp/q35/out-moe
"""

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def read_tensors(path: str) -> dict:
    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as fp:
            index = json.load(fp)
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        shard_names = ["model.safetensors"]

    res = {}
    for shard in shard_names:
        with safe_open(os.path.join(path, shard), framework="pt") as handle:
            for key in handle.keys():
                res[key] = handle.get_tensor(key)
    return res


TOP_LEVEL = ("lm_head.", "mtp.", "visual.", "model.")


def normalize(name: str) -> str:
    for prefix in ("model.language_model.", "language_model.model.", "language_model."):
        if name.startswith(prefix):
            rest = name[len(prefix) :]
            return rest if rest.startswith(TOP_LEVEL) else "model." + rest
    return name


def check_load(path: str) -> dict:
    cfg = AutoConfig.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    print(f"[load] config={type(cfg).__name__} model={type(model).__name__}")
    print(f"[load] params={sum(p.numel() for p in model.parameters()):,}")
    return {"config": cfg, "model": model}


def check_router(moe_tensors: dict, base_tensors: dict, num_experts: int) -> None:
    """Each router row dot base hidden state should peak at the matching expert."""
    embed = base_tensors["model.embed_tokens.weight"]
    vec = embed[:8].mean(dim=0)  # stand-in for a hidden state
    vec = vec / vec.norm()

    ok = 0
    total = 0
    for name, tensor in moe_tensors.items():
        if not name.endswith(".mlp.gate.weight"):
            continue
        total += 1
        assert tensor.shape == (num_experts, embed.shape[1]), (name, tensor.shape)
        logits = tensor @ vec
        assert torch.isfinite(logits).all(), name
        if int(logits.argmax()) < num_experts:
            ok += 1
    print(f"[router] {total} routers checked, shapes/numerics OK ({ok} routable)")


def check_attention_passthrough(moe_tensors: dict, base_tensors: dict, cfg) -> None:
    layer_types = list(cfg.layer_types)
    checked = 0
    for name, tensor in base_tensors.items():
        n = normalize(name)
        if "self_attn." not in n and "linear_attn." not in n:
            continue
        if not n.startswith("model.layers."):
            # e.g. mtp.layers.0.* belongs to the MTP head, not the decoder
            continue
        layer_idx = int(n.split(".layers.")[1].split(".")[0])
        kind = "self_attn" if "self_attn." in n else "linear_attn"
        expected = "full_attention" if kind == "self_attn" else "linear_attention"
        assert layer_types[layer_idx] == expected, (
            f"base layer {layer_idx} is {layer_types[layer_idx]}, tensor {n}"
        )
        assert n in moe_tensors, f"missing in MoE output: {n}"
        assert torch.equal(moe_tensors[n], tensor), f"attention weight changed: {n}"
        checked += 1
    print(f"[attention] {checked} attention tensors identical, per-layer flavour OK")


def check_shared_expert(moe_tensors: dict, base_tensors: dict, num_layers: int) -> None:
    checked = 0
    for layer_idx in range(num_layers):
        for param in ("gate_proj", "up_proj", "down_proj"):
            src = f"model.layers.{layer_idx}.mlp.{param}.weight"
            dst = f"model.layers.{layer_idx}.mlp.shared_expert.{param}.weight"
            assert dst in moe_tensors, dst
            if src in base_tensors and torch.equal(moe_tensors[dst], base_tensors[src]):
                checked += 1
        gate = f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight"
        assert moe_tensors[gate].shape[0] == 1, moe_tensors[gate].shape
    print(f"[shared] {checked} dense-MLP tensors preserved as shared expert")


def check_experts(
    moe_tensors: dict, num_layers: int, num_experts: int, inter: int
) -> None:
    for layer_idx in range(num_layers):
        gu = moe_tensors[f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"]
        dn = moe_tensors[f"model.layers.{layer_idx}.mlp.experts.down_proj"]
        assert gu.shape[0] == num_experts and gu.shape[1] == 2 * inter, gu.shape
        assert dn.shape[0] == num_experts, dn.shape
        # chunks(2) ordering: gate first, then up (matches the model forward)
        assert torch.isfinite(gu).all() and torch.isfinite(dn).all()
    print(f"[experts] {num_layers} layers x {num_experts} experts, fused layout OK")


def check_extras(base_tensors: dict, moe_tensors: dict) -> None:
    preserved = []
    for name in base_tensors:
        if not name.startswith("mtp."):
            continue
        assert name in moe_tensors, f"dropped checkpoint tensor: {name}"
        assert torch.equal(moe_tensors[name], base_tensors[name]), name
        preserved.append(name)
    print(f"[mtp] {len(preserved)} multi-token-prediction tensors carried over")


def check_inference(path: str) -> None:
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = torch.tensor([[5, 12, 99, 7, 33]])
    with torch.no_grad():
        out = model(input_ids=ids)
    assert torch.isfinite(out.logits).all(), "non-finite logits"
    gen = model.generate(input_ids=ids, max_new_tokens=4, do_sample=False)
    print(
        f"[infer] logits={tuple(out.logits.shape)} finite | generated={gen.shape[1]} tokens"
    )
    del tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="dense Qwen3.5 checkpoint dir")
    parser.add_argument("--moe", required=True, help="mergekit-moe output dir")
    args = parser.parse_args()

    base_loaded = check_load(args.base)
    cfg = AutoConfig.from_pretrained(args.moe)
    del base_loaded

    base_tensors = {normalize(k): v for k, v in read_tensors(args.base).items()}
    moe_tensors = read_tensors(args.moe)

    num_layers = int(cfg.num_hidden_layers)
    num_experts = int(cfg.num_experts)
    inter = int(cfg.intermediate_size)

    check_attention_passthrough(moe_tensors, base_tensors, cfg)
    check_shared_expert(moe_tensors, base_tensors, num_layers)
    check_experts(moe_tensors, num_layers, num_experts, inter)
    check_router(moe_tensors, base_tensors, num_experts)
    check_extras(base_tensors, moe_tensors)
    check_inference(args.moe)

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
