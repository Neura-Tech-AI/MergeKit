# Qwen3.5 (hybrid attention) → MoE

This document explains how to turn **hybrid-attention Qwen3.5 checkpoints** — such as
[`sartajbhuvaji/unfuse-Qwen3.8-27B-text`](https://huggingface.co/sartajbhuvaji/unfuse-Qwen3.8-27B-text)
— into a Mixture-of-Experts model with `mergekit-moe`.

Upstream mergekit only ships `qwen3` / `qwen2` MoE output architectures. Qwen3.5 is a
different beast, and every generic code path in mergekit breaks on it in a different way.
This repository adds a dedicated `Qwen3.5 MoE` output architecture plus a mergekit
architecture definition, so **any number of Qwen3.5 checkpoints can be merged**.

## Why Qwen3.5 is not "just Qwen3"

Looking at `config.json` of `unfuse-Qwen3.8-27B-text`:

```json
{
  "model_type": "qwen3_5_text",
  "architectures": ["Qwen3_5ForCausalLM"],
  "hidden_size": 5120,
  "intermediate_size": 17408,
  "num_hidden_layers": 64,
  "full_attention_interval": 4,
  "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention", "..."],
  "mtp_num_hidden_layers": 1,
  "vocab_size": 248320
}
```

Four things differ from Qwen3, and each one breaks a naive merge:

| # | Qwen3.5 behaviour | What breaks without special handling |
|---|---|---|
| 1 | **Hybrid attention** — 48 of 64 layers are gated delta-net `linear_attn.*` (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `conv1d`, `A_log`, `dt_bias`, `norm`, `out_proj`); every 4th layer is full `self_attn.*` (`q/k/v/o_proj`, `q_norm`, `k_norm`) | A single layer template cannot declare both flavours as required. With `optional: true` you get silent key drops; the model then loads with **newly initialised random attention weights**. |
| 2 | **`mtp.*` multi-token-prediction head** (15 tensors: `mtp.fc`, `mtp.pre_fc_norm_*`, `mtp.layers.0.*`, `mtp.norm`) | These are *not* in the `Qwen3_5ForCausalLM` state dict. Any "copy what the model owns" loop drops them — the MTP head is destroyed. |
| 3 | **VLM checkpoints** store the text stack under `model.language_model.*` | Names like `model.language_model.layers.3.self_attn.q_proj.weight` never match `model.layers.3.self_attn.q_proj.weight`. |
| 4 | **`Qwen3_5MoeTextConfig` is a strict dataclass** | Passing through mismatched kwargs (`vision_config`, wrong `model_type`) raises at load time. |

The `unfuse-*` checkpoint is already vision-removed and text-only, so it uses bare
`model.*` names — but the code handles the VLM spelling too, because upstream
`Qwen/Qwen3.8-27B` (and other "unfused" variants) ship that way.

## What the implementation does

`mergekit/moe/qwen3_5.py` adds the `Qwen3.5 MoE` output architecture:

1. **Per-layer attention passthrough.** For each layer it reads `config.layer_types` and
   copies *only* that layer's attention flavour, bit-identical from the base model.
   No required-vs-optional guessing, no random re-init.
2. **Routed experts.** Each configured expert model contributes `gate_proj` + `up_proj`
   fused into `experts.gate_up_proj` (shape `[num_experts, 2*intermediate, hidden]`)
   and `down_proj` into `experts.down_proj` (`[num_experts, hidden, intermediate]`) —
   exactly the layout `Qwen3_5MoeExperts.forward` expects (`chunk(2)` → gate, up).
3. **Dense MLP becomes the shared expert.** Qwen3.5-MoE is built with an always-on
   shared expert plus a learned sigmoid gate; the base model's dense `mlp.*` is copied to
   `mlp.shared_expert.*`, and `mlp.shared_expert_gate.weight` is initialised with
   `nn.Linear`'s default `U(-1/√fan_in, 1/√fan_in)`.
4. **Router.** `mlp.gate.weight` rows come from mergekit's prompt-based gate generation
   (`hidden`, `cheap_embed`, `random`, `uniform_random`).
5. **Extras survive.** Anything the text model does not own (`mtp.*`, `model.visual.*`,
   other towers) is carried over verbatim, and a `mergekit_moe.json` manifest lists what
   was preserved.
6. **Strict config output.** Emits `model_type: qwen3_5_moe_text` /
   `architectures: ["Qwen3_5MoeForCausalLM"]` with `vision_config` stripped.

`mergekit/_data/architectures/qwen3_5.json` registers the dense architecture, so
**regular dense merges** (`mergekit-yaml` with `linear`, `slerp`, `ties`, `dare_ties`,
`model_stock`, `passthrough`, ...) also work on Qwen3.5 checkpoints.

## Usage

```yaml
# moe.yml
base_model: sartajbhuvaji/unfuse-Qwen3.8-27B-text
experts:
  - source_model: sartajbhuvaji/unfuse-Qwen3.8-27B-text
    positive_prompts:
      - "Write a Python function that parses JSON"
      - "Debug this stack trace"
  - source_model: my-org/qwen38-finetune-reasoning
    positive_prompts:
      - "Prove that the square root of 2 is irrational"
      - "Solve this step by step"
  - source_model: my-org/qwen38-finetune-creative
    positive_prompts:
      - "Write a short story about a lighthouse"
      - "Describe this scene poetically"
gate_mode: hidden
experts_per_token: 2
dtype: bfloat16
```

```bash
mergekit-moe moe.yml ./qwen38-moe --copy-tokenizer
```

Roughly 112 GB of disk is needed for three 27B experts plus the output; use `--lazy-unpickle`
and a big `--transformers-cache` if downloads are the bottleneck.

Prompt tips for useful routing:

* give **at least 3–4 distinct prompts per expert**, in the style you expect at inference;
* keep prompts *semantically far apart* — mergekit warns when gate vectors are degenerate
  (identical prompts produce a model where one expert is never selected);
* `gate_mode: hidden` runs a real forward pass per prompt (slow, best quality);
  `gate_mode: cheap_embed` is far cheaper and usually good enough for distinct domains.

## Verifying the result

```bash
python tools/demo/verify_qwen35_moe.py --base <dense checkpoint> --moe <output dir>
```

On the tiny offline fixtures it reports:

```
[attention] 33 attention tensors identical, per-layer flavour OK
[shared]    12 dense-MLP tensors preserved as shared expert
[experts]   4 layers x 2 experts, fused layout OK
[router]    4 routers checked, shapes/numerics OK
[mtp]       15 multi-token-prediction tensors carried over
[infer]     logits=(1, 5, 1024) finite | generated=9 tokens
ALL CHECKS PASSED
```

The output loads cleanly with `AutoModelForCausalLM` — no `MISSING`, no `MISMATCH`.

## Offline fixtures

Because this sandbox cannot reach `huggingface.co`, `tools/demo/make_tiny_qwen35.py`
builds checkpoints with the *same tensor layout* as the real model (hybrid layers, MTP
head, optional VLM prefix) at ~4M parameters:

```bash
python tools/demo/make_tiny_qwen35.py --out /tmp/q35
mergekit-moe /tmp/q35/moe.yml /tmp/q35/out-moe --copy-tokenizer --device cpu
python tools/demo/verify_qwen35_moe.py --base /tmp/q35/model-a --moe /tmp/q35/out-moe
```

## Tests

```bash
python -m pytest tests/test_moe_qwen3_5.py -q
```

17 tests cover name normalisation, layer-type mapping, capability checks, output config
fidelity, the end-to-end conversion, and the VLM-prefixed variant.

## Limitations

* **MTP head is copied, not merged.** All experts' MTP tensors are identical (they come
  from the base model), which is correct: MTP is a draft head, not an expert.
* `shared_experts:` in the YAML supports at most one entry — Qwen3.5-MoE has exactly one,
  and the base model's dense MLP already fills it.
* Expert count should ideally be a power of two for `llama.cpp` compatibility
  (mergekit warns otherwise).
* This produces a **untrained** MoE. Routing is prompt-based; quality only matches the
  dense models if the experts really are specialised. Fine-tune after merging for best
  results.
