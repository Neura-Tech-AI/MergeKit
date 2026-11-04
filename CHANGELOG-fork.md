# mixtureforge — changes relative to upstream mergekit

mixtureforge is a fork of [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit)
at `0f1423ecf0f06f46f17b045b333b6c445705390e` (v0.1.5). Upstream work is
LGPL-3.0-only and stays that way here.

## Rename

The Python package, distribution name and console scripts are `mixtureforge`
instead of `mergekit`, so this fork can be installed next to upstream without
either shadowing the other.

| before | after |
|---|---|
| `import mergekit` | `import mixtureforge` |
| `pip install mergekit` | `pip install -e .` (or `./install.sh`) |
| `mergekit-yaml` | `mixtureforge-yaml` |
| `mergekit-moe` | `mixtureforge-moe` |
| `mergekit-multi` | `mixtureforge-multi` |
| `mergekit-evolve` | `mixtureforge-evolve` |
| `mergekit-extract-lora` | `mixtureforge-extract-lora` |
| `mergekit-layershuffle` | `mixtureforge-layershuffle` |
| `mergekit-tokensurgeon` | `mixtureforge-tokensurgeon` |
| `mergekit-legacy` | `mixtureforge-legacy` |
| `mergekit-pytorch` | `mixtureforge-pytorch` |
| `bakllama` | `bakllama` (unchanged) |
| `mergekit_config.yml` (output) | `mixtureforge_config.yml` |
| `mergekit_moe_config.yml` (output) | `mixtureforge_moe_config.yml` |
| `mergekit_moe.json` (output) | `mixtureforge_moe.json` |

The old output filenames are still recognised: merging a model that was produced
by upstream `mergekit` will not copy its stale `mergekit_*.yml` into the new
output.

## New: Qwen3.5 MoE output architecture

`mixtureforge-moe` can convert dense Qwen3.5 (hybrid linear/full attention)
checkpoints into a standard `qwen3_5_moe_text` model.

- 64-layer aware: each layer gets the attention flavour `config.layer_types`
  says it should have, so a linear layer never receives `self_attn` weights.
- `mtp.*` and vision tensors are carried over verbatim instead of dropped.
- VLM checkpoints stored under `model.language_model.*` are accepted.
- The dense MLP of the base model becomes the always-on shared expert
  (`shared_expert` + `shared_expert_gate`), matching how Qwen3.5-MoE is built.
- Inputs must be **dense** Qwen3.5 models; an already-sparse `qwen3_5_moe*`
  checkpoint has no dense MLP to lift and is rejected with a clear message.

See [docs/qwen3_5_moe.md](docs/qwen3_5_moe.md).

## New: auxiliary files survive a MoE merge

`mergekit-moe` wrote only `config.json`, the weights and (optionally) a
reserialised tokenizer. Everything else the checkpoint needs was silently
dropped: `generation_config.json`, `chat_template.jinja`, `vocab.json`,
`merges.txt`, `.gitattributes`, and the modality markers the Qwen tokenizer
still references. MoE merges now copy those files, byte-identical, from the base
model — matching what dense merges already did through `tagalong_files`.

## New: MoE merges write a model card

Dense merges wrote `README.md`; the MoE path carried a
`# TODO: generate a README.md as well`. It now writes a card listing the output
architecture, `gate_mode`, `experts_per_token`, every expert with its prompts,
and the source YAML.

## Fix: `MergeOptions` flags were silently ignored

`add_merge_options` built its dict from `kwargs` only. Options left at their
click default never appear there — they live in `ctx.params` — so a flag was
parsed, discarded, and the default won. `--no-copy-tokenizer` did nothing. All
eight entrypoints that use the decorator were affected, dense merges included.

## Tooling

- `tools/demo/make_27b_clone.py` — a shape-faithful, width-shrunk clone of the
  published 27B Qwen3.5 checkpoint (64 layers, 48 linear + 16 full attention,
  `attn_output_gate`, GatedDeltaNet, untied embeddings, `mtp.*` head, real
  byte-level BPE tokenizer). Used to exercise the 27B code path without the
  49 GB download.
- `tools/demo/preflight_moe.py` — validates a MoE config from config files
  alone: input types, architecture acceptance, shape agreement, parameter
  count, and free disk/RAM against what the merge needs.
- `install.sh` — the `pip install mergekit` equivalent for this fork.
