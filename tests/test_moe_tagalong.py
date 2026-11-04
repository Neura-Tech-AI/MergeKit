# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""`mergekit-moe` must carry over every file a checkpoint needs, like dense merges.

Dense merges honour `tagalong_files` from the architecture JSON, but the MoE path
originally wrote only config + weights (+ an optional reserialised tokenizer),
silently dropping `generation_config.json`, `chat_template.jinja`, `vocab.json`,
`merges.txt` and preprocessors. These tests lock in the fix.
"""

import os
import pathlib
import tempfile

import pytest

from mergekit.moe import ALL_OUTPUT_ARCHITECTURES
from mergekit.scripts.moe import (
    MOE_SKIP_FILES,
    copy_tagalong_files,
)


@pytest.fixture
def donor(tmp_path):
    """A directory shaped like a real Qwen3.5 text checkpoint."""
    files = {
        "config.json": "{}",
        "model.safetensors": "weights",
        "model-00001-of-00002.safetensors": "shard",
        "model.safetensors.index.json": "{}",
        "tokenizer.json": '{"v":1}',
        "tokenizer_config.json": "{}",
        "vocab.json": "{}",
        "merges.txt": "#version",
        "chat_template.jinja": "{{ m }}",
        "generation_config.json": '{"eos":1}',
        "special_tokens_map.json": "{}",
        "architecture.svg": "<svg/>",
        "README.md": "# upstream card",
        "mergekit_moe.json": "{}",
        ".gitattributes": "* filter=lfs",
    }
    for name, body in files.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


def test_copies_helper_files(donor):
    with tempfile.TemporaryDirectory() as out:
        copied = copy_tagalong_files(str(donor), out)

        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
            "generation_config.json",
            "special_tokens_map.json",
            "architecture.svg",
            ".gitattributes",
        ):
            assert name in copied, f"{name} not copied"
            assert os.path.exists(os.path.join(out, name))

        # byte-identical, not reserialised
        assert (donor / "generation_config.json").read_bytes() == open(
            os.path.join(out, "generation_config.json"), "rb"
        ).read()


def test_skips_weights_and_generated_files(donor):
    with tempfile.TemporaryDirectory() as out:
        copied = copy_tagalong_files(str(donor), out)

        for name in (
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "README.md",
            "mergekit_moe.json",
        ):
            assert name not in copied, f"{name} should not be copied"
            assert not os.path.exists(os.path.join(out, name))


def test_does_not_overwrite_merge_output(donor):
    with tempfile.TemporaryDirectory() as out:
        target = os.path.join(out, "tokenizer_config.json")
        with open(target, "w", encoding="utf-8") as fp:
            fp.write('{"written":"by-merge"}')

        copied = copy_tagalong_files(str(donor), out)

        assert "tokenizer_config.json" not in copied
        with open(target, encoding="utf-8") as fp:
            assert fp.read() == '{"written":"by-merge"}'


def test_extra_files_argument(donor):
    (donor / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (donor / "normalizer.json").write_text("{}", encoding="utf-8")

    with tempfile.TemporaryDirectory() as out:
        copied = copy_tagalong_files(
            str(donor), out, extra_files=["preprocessor_config.json"]
        )
        assert "preprocessor_config.json" in copied
        # .json is in the default set anyway, so this is a no-op extension
        assert "normalizer.json" in copied


def test_missing_donor_is_safe():
    with tempfile.TemporaryDirectory() as out:
        assert copy_tagalong_files("/nonexistent/path", out) == []
        assert copy_tagalong_files("", out) == []


def test_skip_list_contains_generated_files():
    for name in ("config.json", "readme.md", "mergekit_moe.json"):
        assert name in MOE_SKIP_FILES


# --------------------------------------------------------------------------- #
# architecture contract
# --------------------------------------------------------------------------- #
def test_every_moe_arch_exposes_tagalong_files():
    for arch in ALL_OUTPUT_ARCHITECTURES:
        files = arch.tagalong_files()
        assert isinstance(files, list)


def test_qwen3_5_declares_tokenizer_files():
    from mergekit.moe.qwen3_5 import Qwen3_5MoE

    files = Qwen3_5MoE().tagalong_files()
    for expected in (
        "generation_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ):
        assert expected in files


def test_other_archs_still_work_without_tagalong():
    """The default implementation must not require overrides."""
    from mergekit.moe.mixtral import MixtralMoE

    assert MixtralMoE().tagalong_files() == []


# --------------------------------------------------------------------------- #
# end to end: a merge must preserve the full file set
# --------------------------------------------------------------------------- #
def test_moe_merge_preserves_file_set():
    import json
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "demo"))
    from make_tiny_qwen35 import build

    from mergekit.common import ModelReference
    from mergekit.moe.config import Expert, MoEMergeConfig
    from mergekit.options import MergeOptions
    from mergekit.scripts.moe import build as moe_build

    with tempfile.TemporaryDirectory() as tmp:
        src = build(tmp, "src", seed=0, vlm_prefix=False, with_mtp=True)
        extras = {
            "generation_config.json": '{"do_sample": true}',
            "chat_template.jinja": "{{ messages }}",
            "merges.txt": "#version: 0.2",
            "vocab.json": "{}",
        }
        for name, body in extras.items():
            with open(os.path.join(src, name), "w", encoding="utf-8") as fp:
                fp.write(body)

        ref = ModelReference.parse(src)
        out = os.path.join(tmp, "moe")
        moe_build(
            MoEMergeConfig(
                base_model=ref,
                experts=[
                    Expert(source_model=ref, positive_prompts=["math"]),
                    Expert(source_model=ref, positive_prompts=["code"]),
                ],
                gate_mode="random",
                experts_per_token=2,
                dtype="float32",
            ),
            out,
            MergeOptions(device="cpu", copy_tokenizer=True),
            load_in_4bit=False,
            load_in_8bit=False,
            device="cpu",
        )

        produced = set(os.listdir(out))
        for name in extras:
            assert name in produced, f"{name} was dropped by the merge"

        # and the merged config is still the MoE one
        with open(os.path.join(out, "config.json"), encoding="utf-8") as fp:
            cfg = json.load(fp)
        assert cfg["model_type"] == "qwen3_5_moe_text"
        assert cfg["num_experts"] == 2


# --- input-architecture restrictions (parity with the other MoE archs) ------


def test_qwen3_5_moe_rejects_sparse_inputs():
    """A Qwen3.5-MoE checkpoint has no dense MLP, so it cannot be an expert.

    Every other MoE architecture accepts dense inputs only; Qwen3.5 must match.
    """
    from mergekit.moe.qwen3_5 import SUPPORTED_MODEL_TYPES

    assert "qwen3_5_moe_text" not in SUPPORTED_MODEL_TYPES
    assert "qwen3_5_moe" not in SUPPORTED_MODEL_TYPES


def test_dense_input_restriction_matches_sibling_archs():
    """Siblings reject wrong `model_type` inline; Qwen3.5 must do the same."""
    from mergekit.moe.config import Expert, MoEMergeConfig
    from mergekit.moe.qwen3_5 import Qwen3_5MoE

    arch = Qwen3_5MoE()
    cfg = MoEMergeConfig(
        base_model="org/base",
        experts=[Expert(source_model="org/a")],
        gate_mode="hidden",
        experts_per_token=1,
    )

    # every input, including the base, is checked (as in mixtral/deepseek/qwen)
    checked = []
    src = pathlib.Path("mergekit/moe/qwen3_5.py").read_text()
    body = src[src.index("def supports_config") : src.index("def _generate_config")]
    assert "config.base_model" in body, "base model must be validated too"
    assert "config.shared_experts" in body, "shared experts must be validated too"
    assert str(cfg.base_model) == "org/base"
    assert isinstance(arch.name(), str)


# --- README.md model card (dense-path parity) -------------------------------


def test_card_template_renders():
    from mergekit.moe.config import Expert, MoEMergeConfig
    from mergekit.scripts.moe import generate_moe_card

    cfg = MoEMergeConfig(
        base_model="org/base",
        experts=[
            Expert(source_model="org/a", positive_prompts=["math"]),
            Expert(source_model="org/b", positive_prompts=["code"]),
        ],
        gate_mode="hidden",
        experts_per_token=2,
    )
    with tempfile.TemporaryDirectory() as tmp:
        card = generate_moe_card(
            cfg, "base_model: org/base\n", "Qwen3.5 MoE", tmp, name="demo"
        )
        written = pathlib.Path(tmp, "README.md")
        assert written.is_file()
        assert written.read_text() == card
        assert "mixture-of-experts" in card
        assert "expert 0" in card and "math" in card
        assert "shared 0" not in card
        assert "gate_mode: hidden" in card


def test_card_lists_shared_experts():
    from mergekit.moe.config import Expert, MoEMergeConfig
    from mergekit.scripts.moe import generate_moe_card

    cfg = MoEMergeConfig(
        base_model="org/base",
        experts=[Expert(source_model="org/a")],
        shared_experts=[Expert(source_model="org/shared")],
        gate_mode="hidden",
        experts_per_token=1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        card = generate_moe_card(cfg, "", "Qwen3.5 MoE", tmp)
        assert "org/shared" in card
        assert "always active" in card


def test_card_failure_does_not_break_merge():
    """A bad card path must be logged, never raised out of build()."""
    from mergekit.moe.config import Expert, MoEMergeConfig
    from mergekit.scripts.moe import generate_moe_card

    cfg = MoEMergeConfig(
        base_model="org/base",
        experts=[Expert(source_model="org/a")],
        gate_mode="hidden",
        experts_per_token=1,
    )
    missing = "/nonexistent-dir-xyz"
    with pytest.raises(OSError):
        generate_moe_card(cfg, "", "Qwen3.5 MoE", missing)

    src = pathlib.Path("mergekit/scripts/moe.py").read_text()
    body = src[src.index("if merge_options.write_model_card") :]
    body = body[body.index("try:") :]
    assert "except Exception" in body
    assert "Could not generate model card" in body
