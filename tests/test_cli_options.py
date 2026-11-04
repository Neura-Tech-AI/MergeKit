# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Every CLI flag must reach `MergeOptions`.

`add_merge_options` used to build its dict only from `kwargs`, i.e. only from
options the caller passed explicitly. Anything left at its click default lived
in `ctx.params` instead, so a flag like `--no-copy-tokenizer` was parsed, then
silently dropped, and the `MergeOptions` default won. Every one of the eight
entrypoints that use the decorator was affected, dense merges included.
"""

import pytest
from click.testing import CliRunner

click = pytest.importorskip("click")

from mixtureforge.options import MergeOptions, add_merge_options  # noqa: E402


def _make_command():
    @click.command()
    @add_merge_options
    def cmd(merge_options):
        click.echo(
            f"copy_tokenizer={merge_options.copy_tokenizer} "
            f"clone_tensors={merge_options.clone_tensors} "
            f"lazy_unpickle={merge_options.lazy_unpickle} "
            f"out_shard_size={merge_options.out_shard_size} "
            f"write_model_card={merge_options.write_model_card} "
            f"random_seed={merge_options.random_seed}"
        )

    return cmd


@pytest.mark.parametrize(
    "flag,expected",
    [
        ("--no-copy-tokenizer", "copy_tokenizer=False"),
        ("--copy-tokenizer", "copy_tokenizer=True"),
        ("--clone-tensors", "clone_tensors=True"),
        ("--lazy-unpickle", "lazy_unpickle=True"),
        ("--no-write-model-card", "write_model_card=False"),
        ("--random-seed 42", "random_seed=42"),
        ("--out-shard-size 2b", "out_shard_size=2000000000"),
    ],
)
def test_flag_reaches_merge_options(flag, expected):
    result = CliRunner().invoke(_make_command(), flag.split())
    assert result.exit_code == 0, result.output
    assert expected in result.output, result.output


def test_defaults_match_merge_options():
    """An unflagged run must produce exactly MergeOptions()."""
    defaults = MergeOptions()
    result = CliRunner().invoke(_make_command(), [])
    assert result.exit_code == 0, result.output
    assert f"copy_tokenizer={defaults.copy_tokenizer}" in result.output
    assert f"clone_tensors={defaults.clone_tensors}" in result.output
    assert f"lazy_unpickle={defaults.lazy_unpickle}" in result.output
    assert f"out_shard_size={defaults.out_shard_size}" in result.output
    assert f"write_model_card={defaults.write_model_card}" in result.output


def test_non_default_flag_still_works():
    result = CliRunner().invoke(
        _make_command(), ["--no-copy-tokenizer", "--clone-tensors"]
    )
    assert result.exit_code == 0, result.output
    assert "copy_tokenizer=False" in result.output
    assert "clone_tensors=True" in result.output
