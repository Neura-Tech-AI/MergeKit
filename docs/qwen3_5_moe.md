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

## Gate modes — which one to use

`mergekit-moe` builds the router from your `positive_prompts`. The mode decides
*where* the gate vectors come from. All modes were measured on the same trained
specialists (`[math] -> expert0`, `[code] -> expert1`), reporting the cosine
between the two experts' gate vectors — **lower is better**, because a gate has to
tell them apart:

| `gate_mode`    | per-layer | source of the gate vector | math↔code cosine |
|----------------|-----------|---------------------------|------------------|
| `hidden`       | yes       | `output_hidden_states[i]` (pre-norm residual stream) | `[0.04, -0.10, -0.15, -0.13]` |
| `hidden_last`  | yes       | same, last token only | `[0.04, -0.10, -0.15, -0.13]` |
| `hidden_avg`   | yes       | same, averaged over the prompt | `[0.33, 0.34, 0.35, 0.35]` |
| `router_input` | yes       | the tensor each MoE block **actually receives** | `[0.57, 0.59, 0.48, 0.45]` |
| `cheap_embed`  | no        | mean token embedding, one vector for all layers | — |
| `random`       | —         | random (use for training afterwards) | — |

`hidden` is the default and gives the sharpest separation of the prompt-driven
modes, so it is the right choice for most merges.

### The subtlety behind `router_input`

`hidden` reads `output_hidden_states[i]`, the *pre-norm residual stream* entering
decoder layer `i`. The router is not fed that: it gets
`hidden + attn(post_attention_layernorm(hidden))`, i.e. the post-attention
normalised state. On layer 0 the two spaces differ a lot (cosine between a
gate vector and its query is `0.16` for `hidden` vs `0.68` for `router_input`).
On deeper layers they converge, which is why `hidden` still works well in
practice — but if routing looks unstable on a model with large per-layer drift,
try `gate_mode: router_input`.

### Practical prompt tips

* **3–4 distinct prompts per expert**, in the phrasing you expect at inference.
* Keep prompts semantically far apart — mergekit prints
  `degenerate routing parameters` and `experts will be underutilized` when they
  are not.
* Check the separation yourself before a big run:

```python
from mergekit.moe.router import get_gate_params
from mergekit.moe.config import Expert
from mergekit.common import ModelReference

ref = ModelReference.parse("./qwen38-text")
experts = [Expert(source_model=ref, positive_prompts=["code", "debug"]),
           Expert(source_model=ref, positive_prompts=["poem", "story"])]
gv = get_gate_params(ref, tokenizer, experts, mode="hidden", device="cpu")
# gv: (layers, experts, hidden) -> cosine between experts per layer
```

## Which files does the output contain?

`mergekit-moe` writes `config.json`, the weights and the merged config, then —
with `--copy-tokenizer` — the tokenizer **and every auxiliary file the checkpoint
needs**. Dense merges already honoured `tagalong_files` from the architecture
JSON; the MoE path now does the same. For a Qwen3.5 checkpoint the output keeps:

| copied | why it matters |
|---|---|
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json` | the Qwen2 tokenizer, byte-identical |
| `vocab.json`, `merges.txt` | needed by the slow/tokenizers pair and by conversion tooling |
| `chat_template.jinja` | the real chat template (reasoning/tool-call blocks) |
| `generation_config.json` | sampling defaults — easily forgotten, changes output |
| `.gitattributes`, `architecture.svg` | repo hygiene |
| `mtp.*` tensors | inside `model.safetensors`, not a file |

Deliberately **not** copied: `config.json` (rewritten as the MoE config),
`README.md` (regenerated as the model card), `mergekit_moe.json` (generated),
and any `*.safetensors` shard.

The Qwen3.5 tokenizer keeps modality markers like `<|image_pad|>` even in the
text-only checkpoint, which is why shipping the raw tokenizer files instead of
only a reserialised object matters.

## What can be used as an expert?

Only **dense** Qwen3.5 checkpoints (`qwen3_5_text`, `qwen3_5`). Every other MoE
output in mergekit is built by sweeping up a dense `mlp.gate_proj` / `up_proj` /
`down_proj` triple, and a Qwen3.5-MoE checkpoint stores `mlp.experts.gate_up_proj`
instead — there is no dense MLP to lift, so using one as a base or expert fails.
This matches Mixtral (llama/mistral only), DeepSeek (llama/mistral only) and
Qwen (llama/mistral/qwen2 only). Passing a `qwen3_5_moe*` model raises a
specific warning rather than a confusing missing-tensor error.

The output is still a standard `qwen3_5_moe_text` checkpoint that loads and can
itself be merged by the dense path.

## Verified end to end at 27B geometry

The published 27B checkpoint cannot be downloaded in CI, so
`tools/demo/make_27b_clone.py` reproduces its architecture exactly and shrinks
only the widths: **64 layers, 48 linear + 16 full attention** (full attention on
every 4th layer), `attn_output_gate`, GatedDeltaNet hyper-parameters, untied
embeddings, partial rotary, mRoPE and an `mtp.*` head.

A real `mergekit-moe` run over three such clones produced a checkpoint with:

| check | result |
|---|---|
| tensor names vs a real 27B MoE model | **0 missing**, 0 rank mismatches |
| `experts.gate_up_proj` | `(3, 2*inter, hidden)`, fused gate+up, as in the real checkpoint |
| `experts.down_proj` | `(3, hidden, inter)` |
| `gate.weight` | `(3, hidden)` |
| `shared_expert_gate.weight` | `(1, hidden)` |
| `layer_types` | unchanged through the conversion |
| loads + forward + `generate` | finite logits, token ids in range |
| tokenizer | encodes and round-trips identically to the source |

`tests/test_moe_real_shape.py` locks all of that in.

## Check the run will fit, before downloading 150 GB

```bash
python tools/demo/preflight_moe.py moe.yml --out-dir ./merged
```

It reads only config files and reports input model types, whether the output
architecture accepts the config, shape agreement (`vocab_size`, `hidden_size`,
`layer_types`), the resulting parameter count, and free disk/RAM against what
the merge actually needs. Exit code 1 means the run would fail. For the 27B:

| experts | output | disk (inputs + output) | gate-extraction RAM |
|---|---|---|---|
| 2 | 73.1B / 146 GB | ~244 GB | ~49 GB |
| 3 | 97.5B / 195 GB | ~341 GB | ~49 GB |
| 4 | 121.8B / 244 GB | ~439 GB | ~49 GB |

`gate_mode: hidden` loads one complete dense model in bf16 to read its hidden
states; the merging itself only holds one layer's expert MLPs at a time.

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

## Train → merge → compare: does the output actually survive?

`tools/demo/train_and_test_qwen35.py` does the full experiment end to end:

```bash
python tools/demo/train_and_test_qwen35.py --out /tmp/q35x --steps 300
```

It builds a tiny Qwen3.5 base, **fine-tunes three specialists** on three disjoint
synthetic tasks (`math`, `code`, `prose` — each a different bijection over the same
token space), converts them to MoE, and then compares outputs.

```
=== specialist accuracy (memorised inputs) ===
          own-domain | cross-domain (learned the wrong mapping)
  math  :     100.0% | 3.1%, 6.2%
  code  :     100.0% | 3.1%, 3.1%
  prose :     100.0% | 6.2%, 3.1%

  [PASS] 1-expert MoE is lossless
         max |logit diff| = 0.000e+00, identical greedy tokens = True
  [PASS] 2-expert MoE routes to the right expert in every layer
         B1 router contract 100% (gates written exactly: True) |
         B2 agreement math 69% code 0% |
         B3 base separability [0.56, 0.45, 0.4, 0.42]
  [PASS] 3 experts stored exactly, model runs
         max |expert weight diff| = 0.000e+00 | finite logits = True |
         generated 2 new tokens | mtp tensors = 0
ALL TESTS PASSED
```

What each check proves:

* **1-expert MoE is lossless** — a MoE built from a single expert produces
  **bit-identical logits** to the dense model (diff `0.000e+00`). Attention
  passthrough, fused expert packing, `lm_head` and `mtp.*` all survive, and greedy
  decoding picks the same tokens for all 16 probes.
* **Router contract** — every decoder layer's router, given that layer's real
  input, selects the expert whose prompts match the domain: **100%, every layer**.
  The gate rows written to the checkpoint match the generated vectors (cosine
  `1.0000`).
* **3 experts stored exactly** — expert `k`'s `gate_up_proj` / `down_proj` slices
  equal that expert's dense weights bit for bit (diff `0.000e+00`), the merged
  model loads and generates, and `mtp.*` round-trips.

### The honest caveat about end-to-end agreement

`B2 agreement code 0%` is *not* a merge bug, and it would look the same for
Mixtral. With top-1 routing every layer picks an expert independently, so a token
passes through a mix of experts — necessarily different from the pure `code`
specialist path. `B3` shows the other half of the story: in the base model some
layers cannot separate these two prompts at all (cosine up to `0.99` for the
`code` model's layers), so no gate vector can route them correctly there.

Practical consequence for real models: **give experts genuinely different data**
and **use 3–4 distinct prompts each**. When prompts are too similar, mergekit
warns about degenerate routing and one expert ends up unused.

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
