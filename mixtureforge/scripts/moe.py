# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import logging
import os
import shutil
import sys
from typing import List, Optional, Sequence

import click
import transformers
import yaml

from mixtureforge.merge import MergeOptions
from mixtureforge.moe import ALL_OUTPUT_ARCHITECTURES, MoEOutputArchitecture
from mixtureforge.moe.config import MoEMergeConfig, is_bad_config
from mixtureforge.moe.router import get_gate_params, warn_degenerate_gates
from mixtureforge.options import PrettyPrintHelp, add_merge_options

# Files produced by the merge itself, or unsafe to carry over.
MOE_SKIP_FILES = frozenset(
    {
        "config.json",
        "readme.md",
        "mixtureforge_config.yml",
        "mixtureforge_moe_config.yml",
        "mixtureforge_moe.json",
        # models produced before the rename still carry these
        "mergekit_config.yml",
        "mergekit_moe_config.yml",
        "mergekit_moe.json",
        "model.safetensors.index.json",
    }
)
MOE_SKIP_PREFIXES = ("model-", "model.safetensors", "pytorch_model")

# Everything that makes a checkpoint usable but is NOT stored in config.json:
# the packed tokenizer, special tokens, chat template, generation defaults,
# sentencepiece files, preprocessors, and (for VLM-turned-text models) the
# modality markers the tokenizer still references.
MOE_TAGALONG_EXTS = (
    ".jinja",
    ".json",
    ".model",
    ".txt",
    ".tiktoken",
    ".py",
    ".svg",
    ".md",
    ".gitattributes",
)


def _is_shard_or_weights(basename: str) -> bool:
    return basename.startswith(MOE_SKIP_PREFIXES) or basename.endswith(".safetensors")


def copy_tagalong_files(
    donor_path: str,
    out_path: str,
    extra_files: Optional[Sequence[str]] = None,
) -> List[str]:
    """Copy the non-weight files that a checkpoint needs to be loadable.

    `mixtureforge-moe` only writes config + weights and, optionally, a tokenizer
    reserialised from `save_pretrained`. That drops `generation_config.json`,
    `chat_template.jinja`, `vocab.json`, `merges.txt` and any architecture
    specific extras declared through `tagalong_files` in the architecture JSON.
    Dense merges already copy those; this brings MoE merges to parity.

    Returns the list of copied file names.
    """
    if not donor_path or not os.path.isdir(donor_path):
        return []

    wanted = set(extra_files or []) | set(MOE_TAGALONG_EXTS)
    copied: List[str] = []
    for basename in sorted(os.listdir(donor_path)):
        # case-insensitive: README.md on disk vs "readme.md" in the skip list
        if basename.lower() in MOE_SKIP_FILES or _is_shard_or_weights(basename):
            continue
        if not any(basename.endswith(ext) for ext in wanted):
            continue
        source = os.path.join(donor_path, basename)
        if not os.path.isfile(source):
            continue
        target = os.path.join(out_path, basename)
        if os.path.exists(target):
            # never clobber what the merge just wrote (e.g. tokenizer_config)
            continue
        shutil.copy(source, target)
        copied.append(basename)
    return copied


MOE_CARD_TEMPLATE = """---
base_model: {base_model}
tags:
- mixtureforge
- merge
- mixture-of-experts
library_name: transformers
---

# {name}

This is a Mixture-of-Experts model built with [mixtureforge](https://github.com/arcee-ai/mixtureforge)
from pre-trained dense language models, using the **{architecture}** output architecture.

## Merge Details

Each expert below is a dense model converted into a set of routed experts. The
router was generated with `gate_mode: {gate_mode}` and routes each token to
`{experts_per_token}` expert(s).

| Role | Model | Prompts |
|---|---|---|
{expert_rows}

## Configuration

The following YAML configuration was used to produce this model:

```yaml
{config_yaml}
```
"""


def generate_moe_card(
    config: MoEMergeConfig,
    config_yaml: str,
    architecture_name: str,
    out_path: str,
    name: Optional[str] = None,
) -> str:
    """Write a README.md model card for a MoE merge.

    Upstream mergekit only wrote ``mixtureforge_moe_config.yml`` here; the dense
    merge path additionally emits a README.md (:func:`mixtureforge.card.generate_card`).
    This restores that parity.
    """
    if not name:
        name = "Untitled MoE (1)"

    rows = []
    for idx, expert in enumerate(config.experts):
        prompts = ", ".join(expert.positive_prompts or []) or "-"
        rows.append(f"| expert {idx} | {expert.source_model} | {prompts} |")
    for idx, expert in enumerate(config.shared_experts or []):
        prompts = ", ".join(expert.positive_prompts or []) or "always active"
        rows.append(f"| shared {idx} | {expert.source_model} | {prompts} |")

    card = MOE_CARD_TEMPLATE.format(
        base_model=config.base_model,
        name=name,
        architecture=architecture_name,
        gate_mode=config.gate_mode,
        experts_per_token=config.experts_per_token,
        expert_rows="\n".join(rows),
        config_yaml=config_yaml,
    )
    with open(os.path.join(out_path, "README.md"), "w", encoding="utf-8") as fp:
        fp.write(card)
    return card


def build(
    config: MoEMergeConfig,
    out_path: str,
    merge_options: MergeOptions,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    device: str = "auto",
    allow_all_same: bool = False,
    verbose: bool = False,
    config_source: Optional[str] = None,
):
    if is_bad_config(config, allow_all_same=allow_all_same):
        sys.exit(1)

    base_model = config.base_model
    out_arch = select_output_arch(config, merge_options, verbose=verbose)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model.model.path, revision=base_model.model.revision
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.bos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Getting gate parameters...")
    need_gates = list(config.experts)
    if config.shared_experts:
        has_prompts = any(e.positive_prompts for e in config.shared_experts)
        assert all(
            bool(e.positive_prompts) == has_prompts for e in config.shared_experts
        ), "Must specify prompts for all shared experts or none, not a mix"
        if has_prompts or config.gate_mode in ("random", "uniform_random"):
            need_gates.extend(config.shared_experts)

    gate_vecs = get_gate_params(
        base_model,
        tokenizer,
        need_gates,
        mode=config.gate_mode,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        lazy_unpickle=merge_options.lazy_unpickle,
        trust_remote_code=merge_options.trust_remote_code,
        device=device,
    )
    # gate_vecs: (num_layers, num_experts, hidden_size)
    router_weights = gate_vecs[:, : len(config.experts), :]
    shared_router_weights = gate_vecs[:, len(config.experts) :, :]
    warn_degenerate_gates(gate_vecs)

    out_arch.write_model(
        out_path,
        config,
        merge_options,
        router_weights=[router_weights[i, ...] for i in range(router_weights.shape[0])],
        shared_router_weights=[
            shared_router_weights[i, ...] for i in range(router_weights.shape[0])
        ],
    )

    if merge_options.write_model_card:
        try:
            if config_source is None:
                config_source = yaml.dump(
                    config.model_dump(mode="json", exclude_none=True),
                    sort_keys=False,
                )
            with open(
                os.path.join(out_path, "mixtureforge_moe_config.yml"),
                "w",
                encoding="utf-8",
            ) as fp:
                fp.write(config_source)
            generate_moe_card(
                config,
                config_source,
                architecture_name=out_arch.name(),
                out_path=out_path,
                name=os.path.basename(os.path.abspath(out_path)),
            )
            logging.info("Wrote README.md model card")
        except Exception as exc:  # a failed card must never fail the merge
            logging.warning(f"Could not generate model card: {exc}")

    if merge_options.copy_tokenizer:
        logging.info("Saving tokenizer...")
        tokenizer.save_pretrained(out_path, safe_serialization=True)

        # Carry over everything the tokenizer references but does not save
        # itself: generation defaults, chat template, vocab/merges, modality
        # markers, and any architecture-specific tagalong files.
        donor_path = base_model.model.path
        try:
            donor_path = (
                base_model.local_path(cache_dir=merge_options.transformers_cache)
                or donor_path
            )
        except Exception:
            pass
        copied = copy_tagalong_files(
            donor_path, out_path, extra_files=out_arch.tagalong_files()
        )
        if copied:
            logging.info(f"Copied {len(copied)} auxiliary file(s): {copied}")
        else:
            logging.info("No auxiliary files found to copy.")

    logging.info("Done.")


def select_output_arch(
    config: MoEMergeConfig,
    merge_options: MergeOptions,
    verbose: bool = False,
) -> MoEOutputArchitecture:
    candidates_in = ALL_OUTPUT_ARCHITECTURES
    if config.architecture:
        candidates_in = [
            a
            for a in candidates_in
            if a.name().lower().startswith(config.architecture.lower())
        ]
    if not candidates_in:
        logging.error(
            f"No output architecture found that matches the given architecture: {config.architecture}"
        )
        logging.error("All supported output architectures:")
        for arch in ALL_OUTPUT_ARCHITECTURES:
            logging.error(f"  * {arch.name()}")
        sys.exit(1)

    candidates: List[MoEOutputArchitecture] = []
    for arch in candidates_in:
        if arch.supports_config(
            config, explain=verbose, trust_remote_code=merge_options.trust_remote_code
        ):
            candidates.append(arch)
        else:
            logging.info(f"Output architecture {arch.name()} does not support config.")

    if not candidates:
        logging.error(
            "No output architecture found that is compatible with the given models."
        )

        logging.error("All supported output architectures:")
        for arch in ALL_OUTPUT_ARCHITECTURES:
            logging.error(f"  * {arch.name()}")
        sys.exit(1)

    # for compatibility with older configs, default to Mixtral if available
    for arch in candidates:
        if arch.name() == "Mixtral":
            return arch

    if len(candidates) > 1:
        logging.warning(
            "Multiple output architectures found that are compatible with the given models."
        )
        logging.warning(f"Defaulting to {candidates[0].name()}")
    else:
        logging.info(f"Selected output architecture: {candidates[0].name()}")
    return candidates[0]


@click.command("mixtureforge-moe", cls=PrettyPrintHelp)
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_path", type=click.Path())
@click.option(
    "--load-in-4bit",
    is_flag=True,
    type=bool,
    default=False,
    help="Load model in 4bit for computing hidden states",
)
@click.option(
    "--load-in-8bit",
    is_flag=True,
    type=bool,
    default=False,
    help="Load model in 8bit for computing hidden states",
)
@click.option(
    "--i-understand-this-is-not-useful-without-training",
    type=bool,
    default=False,
    is_flag=True,
    help="Really make the questionable model you want.",
)
@add_merge_options
def main(
    config_path: str,
    out_path: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    i_understand_this_is_not_useful_without_training: bool,
    merge_options: MergeOptions,
):
    """Create a Mixture of Experts model by combining the pretrained weights of multiple models."""
    merge_options.apply_global_options()

    with open(config_path, "r", encoding="utf-8") as file:
        config_source = file.read()

    config = MoEMergeConfig.model_validate(yaml.safe_load(config_source))
    build(
        config,
        out_path=out_path,
        merge_options=merge_options,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        device=merge_options.device,
        allow_all_same=i_understand_this_is_not_useful_without_training,
        verbose=merge_options.verbosity > 0,
        config_source=config_source,
    )


if __name__ == "__main__":
    main()
