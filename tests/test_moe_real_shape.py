# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""End-to-end merge of a shape-faithful clone of the real 27B checkpoint.

The published `sartajbhuvaji/unfuse-Qwen3.8-27B-text` is ~49 GB, far beyond a
normal test machine, so `tools/demo/make_27b_clone.py` reproduces its
architecture exactly -- 64 layers, 48 linear + 16 full attention, output gating,
untied embeddings, an `mtp.*` head -- and only shrinks the widths. These tests
run a real `mergekit-moe` merge over that clone and assert the parts that would
silently break a 27B run: the output geometry, the tokenizer round-trip and the
file manifest.
"""

import json
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "tools" / "demo"))

from make_27b_clone import LAYER_TYPES  # noqa: E402


def test_clone_hybrid_pattern_matches_published_checkpoint():
    """48 linear + 16 full, full attention on every 4th layer."""
    assert len(LAYER_TYPES) == 64
    assert LAYER_TYPES.count("full_attention") == 16
    assert LAYER_TYPES.count("linear_attention") == 48
    assert [i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"] == list(
        range(3, 64, 4)
    )


def test_tokenizer_roundtrip_encodes_words():
    """The fixture must use a real Qwen3.5 byte-level BPE, not a stand-in.

    `Qwen3_5Tokenizer` hardcodes a byte-level BPE pipeline in `__init__`, so a
    WordLevel `tokenizer.json` loads with an unusable vocabulary and encodes
    every prompt to `[]` -- which makes `gate_mode: hidden` fail on empty
    sequences. This guards the fixture against regressing to that.
    """
    import tempfile

    import transformers
    from make_27b_clone import build

    with tempfile.TemporaryDirectory() as tmp:
        path = build(tmp, "tok", seed=0, with_mtp=False)
        tok = transformers.AutoTokenizer.from_pretrained(path)
        ids = tok("write python code")["input_ids"]
        assert len(ids) > 1, f"tokenizer produced {ids}"
        reloaded = transformers.AutoTokenizer.from_pretrained(path)
        assert reloaded("write python code")["input_ids"] == ids


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    """3 dense clones -> one real mergekit-moe run (64 layers each)."""
    import tempfile

    from make_27b_clone import build

    from mergekit.common import ModelReference
    from mergekit.moe.config import Expert, MoEMergeConfig
    from mergekit.options import MergeOptions
    from mergekit.scripts.moe import build as moe_build

    root = tempfile.mkdtemp()
    a = build(root, "dense-a", seed=0)
    b = build(root, "dense-b", seed=1, layer_bias=0.01)
    c = build(root, "dense-c", seed=2, layer_bias=0.02)

    out = str(tmp_path_factory.mktemp("moe"))
    cfg = MoEMergeConfig(
        base_model=ModelReference.parse(a),
        experts=[
            Expert(source_model=ModelReference.parse(a), positive_prompts=["python"]),
            Expert(source_model=ModelReference.parse(b), positive_prompts=["prove"]),
            Expert(source_model=ModelReference.parse(c), positive_prompts=["story"]),
        ],
        shared_experts=[Expert(source_model=ModelReference.parse(a))],
        gate_mode="hidden",
        experts_per_token=2,
        dtype="float32",
    )
    moe_build(
        cfg,
        out,
        MergeOptions(device="cpu", copy_tokenizer=True),
        load_in_4bit=False,
        load_in_8bit=False,
        device="cpu",
    )
    return out


def test_output_geometry_matches_27b_layout(merged):
    cfg = json.loads(pathlib.Path(merged, "config.json").read_text())
    assert cfg["architectures"] == ["Qwen3_5MoeForCausalLM"]
    assert cfg["model_type"] == "qwen3_5_moe_text"
    assert cfg["num_experts"] == 3
    assert cfg["num_experts_per_tok"] == 2
    assert cfg["num_hidden_layers"] == 64
    assert cfg["moe_intermediate_size"] == cfg["intermediate_size"]
    assert cfg["shared_expert_intermediate_size"] == cfg["intermediate_size"]
    # the hybrid pattern must survive the conversion untouched
    assert cfg["layer_types"] == LAYER_TYPES
    assert cfg["layer_types"].count("full_attention") == 16


def test_output_tensor_names_match_real_moe_checkpoint(merged):
    from safetensors import safe_open

    with safe_open(pathlib.Path(merged, "model.safetensors"), "pt") as f:
        names = set(f.keys())

    # the three tensors that define a fused, gated, shared-expert MoE layer
    for required in (
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.layers.0.mlp.shared_expert_gate.weight",
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
    ):
        assert required in names, f"{required} missing from the merged output"

    # dense MLPs must be gone, or the checkpoint is inconsistent
    assert not any(
        n.endswith(".mlp.gate_proj.weight")
        for n in names
        if ".shared_expert." not in n and not n.startswith("mtp.")
    )


def test_merged_model_loads_and_runs(merged):
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(merged)
    ids = tok("write python code")["input_ids"]
    assert len(ids) > 1, "merged tokenizer must encode words"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        merged, dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits
    assert logits.shape == (1, len(ids), model.config.vocab_size)
    assert torch.isfinite(logits).all()


def test_merge_writes_card_and_aux_files(merged):
    files = {p.name for p in pathlib.Path(merged).iterdir()}
    for required in (
        "README.md",
        "config.json",
        "mergekit_moe_config.yml",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        assert required in files, f"{required} missing: {sorted(files)}"
    card = pathlib.Path(merged, "README.md").read_text()
    assert "mixture-of-experts" in card
    assert "Qwen3.5 MoE" in card
