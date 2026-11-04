# Local Development Environment

This checkout is **[mixtureforge](../../)**, a fork of upstream
[`arcee-ai/mergekit`](https://github.com/arcee-ai/mergekit) at commit
`0f1423ecf0f06f46f17b045b333b6c445705390e` (v0.1.5, 2026-10-31).

The package was renamed from `mergekit` to `mixtureforge` so it can be installed
alongside the upstream package without either one shadowing the other.

## Install

```sh
./install.sh              # editable install into ./.venv (recommended)
./install.sh --user       # editable install into the active environment
./install.sh --wheel      # build a wheel and install that (production)
./install.sh --clean      # wipe .venv and start over
```

Or by hand:

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]" pytest
```

The installer ends with a self-check: it imports the package, counts the
architecture definitions, and confirms both console scripts exist.

## The ten console scripts

They are available under `.venv/bin/` and, with `--user`, on your `PATH`:

```sh
.venv/bin/mixtureforge-yaml       .venv/bin/mixtureforge-moe
.venv/bin/mixtureforge-multi      .venv/bin/mixtureforge-evolve
.venv/bin/mixtureforge-extract-lora
.venv/bin/mixtureforge-tokensurgeon  .venv/bin/mixtureforge-layershuffle
.venv/bin/mixtureforge-legacy     .venv/bin/mixtureforge-pytorch
.venv/bin/bakllama
```

## What is already set up in this sandbox

Installed: Python 3.11.2, torch 2.15.0+cu132 (CPU-only runtime, no GPU driver
present), transformers 5.22.1, safetensors 0.8.0, accelerate, peft, pydantic 2.x,
and the rest of `pyproject.toml`.

## Running the test suite

```sh
.venv/bin/python -m pytest tests/ -q
```

## Renamed output files

A merge writes `mixtureforge_config.yml` (dense) or
`mixtureforge_moe_config.yml` + `mixtureforge_moe.json` (MoE). Models produced
by the old `mergekit` name still load, and merging one will **not** copy its
stale `mergekit_*.yml` into the new output — those names are in the skip list.

## Offline demo

```sh
HF_HUB_OFFLINE=1 .venv/bin/python tools/demo/make_27b_clone.py --out /tmp/clone --name a
HF_HUB_OFFLINE=1 .venv/bin/mixtureforge-moe /tmp/clone/moe.yml /tmp/clone/out --copy-tokenizer
```

See `docs/qwen3_5_moe.md` for the full merge/verify walkthrough and
`tools/demo/preflight_moe.py` for the "will this fit on my disk?" check.
