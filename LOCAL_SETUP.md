# Local Development Environment (this sandbox)

This checkout is a full vendored copy of upstream
[`arcee-ai/mergekit`](https://github.com/arcee-ai/mergekit) at commit
`0f1423ecf0f06f46f17b045b333b6c445705390e` (v0.1.5, 2026-10-31).

## What is already set up

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
```

Installed: Python 3.11.2, torch 2.15.0+cu132 (CPU-only runtime, no GPU driver present),
transformers 5.22.1, safetensors 0.8.0, accelerate, peft, pydantic 2.x, and the rest of
`pyproject.toml`.

The ten console scripts are available under `.venv/bin/`:

```sh
.venv/bin/mergekit-yaml     .venv/bin/mergekit-moe        .venv/bin/mergekit-multi
.venv/bin/mergekit-evolve   .venv/bin/mergekit-extract-lora
.venv/bin/mergekit-tokensurgeon  .venv/bin/mergekit-layershuffle
.venv/bin/mergekit-legacy   .venv/bin/mergekit-pytorch    .venv/bin/bakllama
```

## Running the test suite

```sh
.venv/bin/python -m pytest tests/ -q
```

Current result on this machine: **520 passed, 82 skipped** in ~20s.
The skips are GPU-only or Hugging Face-hub-dependent tests.

## Network limitation (important)

The sandbox can only reach `github.com`, `codeload.github.com`, `api.github.com`,
`registry.npmjs.org`, `pypi.org` and `files.pythonhosted.org`. **`huggingface.co` is
blocked**, so nothing can be downloaded from the Hub — any config that references a
Hub model id (`meta-llama/...`, `mistralai/...`) will fail here.

Work around it by building tiny local checkpoints and merging those, which exercises
the exact same code path (graph build, lazy tensor loading, merge methods, sharding,
tokenizer copy, model card generation).

## Verified end-to-end run (fully offline)

```sh
# 1. create two tiny 2-layer Llama checkpoints (~115k params each)
HF_HUB_OFFLINE=1 .venv/bin/python tools/demo/make_tiny_models.py --out /tmp/mk-demo

# 2. merge them
HF_HUB_OFFLINE=1 .venv/bin/mergekit-yaml /tmp/mk-demo/linear.yml \
    /tmp/mk-demo/out-linear --copy-tokenizer
```

Output directory contains `model.safetensors`, `config.json`, `mergekit_config.yml`,
a generated `README.md` model card, and the copied tokenizer.

Numeric check for the `linear` config (`weight: 0.6` / `0.4`):

```
max |0.6*a + 0.4*b - merged| = 1.19e-07   -> OK
```

A `ties` merge (`density: 0.7`, `normalize: true`, base = model-a) also runs clean.

## Notes

- Always pass `HF_HUB_OFFLINE=1` so transformers does not try to phone home.
- Do **not** pass `--cuda` here; there is no NVIDIA driver.
- `.venv/` is git-ignored; it is ~6 GB and must never be committed.
- Disk: ~10 GB free after install. Merging real 7B-class models needs ~16 GB of disk
  per checkpoint, so real-model work needs either the Hub or locally supplied weights.
