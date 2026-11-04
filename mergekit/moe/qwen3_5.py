# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Qwen3.5 (hybrid linear-attention) MoE output architecture.

Qwen3.5 text models are *hybrid*: most decoder layers use gated delta-net
"linear attention" (``linear_attn.*``) while every fourth layer uses regular
full attention (``self_attn.*``). They also ship multi-token-prediction
(``mtp.*``) weights that the text-only transformers classes ignore.

That combination breaks every generic path in mergekit:

* a single layer template cannot list both attention flavours as required,
  because no individual layer has all of them;
* ``mtp.*`` tensors are not part of the CausalLM state dict, so a naive copy
  drops them (losing the MTP head);
* published checkpoints are often VLMs stored under ``model.language_model.*``.

This module handles all three:

* non-MLP tensors are copied from the base model;
* attention tensors are copied per layer according to ``config.layer_types``,
  so a linear layer never receives ``self_attn`` weights and vice versa;
* the dense MLP of each layer becomes the always-on *shared expert* (exactly
  how Qwen3.5-MoE is built), while each configured expert model supplies the
  routed experts;
* every checkpoint tensor that is not part of the text model (``mtp.*``,
  ``model.visual.*``, ...) is carried over verbatim.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import tqdm
import transformers
from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig

from mergekit.architecture import ModelArchitecture, arch_info_for_config
from mergekit.architecture.base import WeightInfo
from mergekit.moe.arch import MoEOutputArchitecture
from mergekit.moe.common import initialize_io, noise_and_scale, select_dtype
from mergekit.moe.config import Expert, MoEMergeConfig
from mergekit.options import MergeOptions

# Checkpoint prefixes that hold a full VLM wrapper around the language model.
LANGUAGE_MODEL_PREFIXES: Tuple[str, ...] = (
    "model.language_model.",
    "language_model.model.",
    "language_model.",
)

EMBEDDING_MARKERS: Tuple[str, ...] = ("embed_tokens", "lm_head")

SUPPORTED_MODEL_TYPES: Tuple[str, ...] = (
    "qwen3_5_text",
    "qwen3_5",
    "qwen3_5_moe_text",
    "qwen3_5_moe",
)

# Config keys that only make sense on a VLM wrapper.
VLM_ONLY_KEYS: Tuple[str, ...] = (
    "vision_config",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)


# Top-level modules that live there already; they never sit under ``model.``.
TOP_LEVEL_MODULES: Tuple[str, ...] = ("lm_head.", "mtp.", "visual.", "model.")


def _inside_language_model(rest: str) -> str:
    """Rebuild a canonical name from what follows a language-model prefix."""
    if rest.startswith(TOP_LEVEL_MODULES):
        return rest
    return "model." + rest


def strip_lm_prefix(name: str) -> str:
    """Map a VLM-prefixed tensor name onto its canonical text-model name.

    ``model.language_model.embed_tokens.weight`` -> ``model.embed_tokens.weight``
    ``model.language_model.lm_head.weight``      -> ``lm_head.weight``
    ``language_model.model.layers.0...``         -> ``model.layers.0...``
    ``mtp.fc.weight``                            -> ``mtp.fc.weight``
    """
    for prefix in LANGUAGE_MODEL_PREFIXES:
        if name.startswith(prefix):
            return _inside_language_model(name[len(prefix) :])
    return name


def candidate_names(name: str) -> List[str]:
    """All checkpoint spellings a canonical text-model tensor may use."""
    res = [name]
    if name.startswith(TOP_LEVEL_MODULES):
        inner = name[len("model.") :] if name.startswith("model.") else name
        res.append("model.language_model." + inner)
        res.append("language_model." + inner)
        res.append("language_model." + name)
    if name.startswith("model."):
        res.append("model.language_model." + name[len("model.") :])
    res.append("model.language_model." + name)
    res.append("language_model." + name)
    # de-duplicate while preserving order
    seen = set()
    out = []
    for candidate in res:
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def is_embedding_name(name: str) -> bool:
    return any(marker in name for marker in EMBEDDING_MARKERS)


def is_mlp_name(name: str) -> bool:
    return ".mlp." in name and not is_embedding_name(name)


def attention_kind(name: str) -> Optional[str]:
    if "self_attn" in name:
        return "self_attn"
    if "linear_attn" in name:
        return "linear_attn"
    return None


def layer_kinds(cfg: transformers.PretrainedConfig, num_layers: int) -> List[str]:
    """Map ``config.layer_types`` onto ``self_attn`` / ``linear_attn``."""
    layer_types = getattr(cfg, "layer_types", None)
    if not layer_types:
        return ["self_attn"] * num_layers
    kinds: List[str] = []
    for idx in range(num_layers):
        layer_type = layer_types[idx] if idx < len(layer_types) else layer_types[-1]
        kinds.append(
            "self_attn" if str(layer_type).endswith("full_attention") else "linear_attn"
        )
    return kinds


def find_tensor(
    loader,
    name: str,
    aliases: Sequence[str] = (),
) -> Optional[torch.Tensor]:
    """Look a tensor up under every plausible checkpoint spelling."""
    seen = set()
    for candidate in (name, *aliases):
        for full_name in candidate_names(candidate):
            if full_name in seen:
                continue
            seen.add(full_name)
            try:
                tensor = loader.get_tensor(full_name)
            except KeyError:
                continue
            if tensor is not None:
                return tensor
    return None


def arch_layer_weights(
    arch: ModelArchitecture, cfg: transformers.PretrainedConfig, layer_idx: int
) -> List[WeightInfo]:
    res: List[WeightInfo] = []
    for module in arch.modules.values():
        res.extend(module.architecture.layer_weights(layer_idx, cfg) or [])
    return res


class Qwen3_5MoE(MoEOutputArchitecture):
    """Turn dense Qwen3.5 checkpoints into a Qwen3.5-MoE checkpoint."""

    def name(self) -> str:
        return "Qwen3.5 MoE"

    # ------------------------------------------------------------------ #
    # capability check
    # ------------------------------------------------------------------ #
    def supports_config(
        self,
        config: MoEMergeConfig,
        explain: bool = False,
        trust_remote_code: bool = False,
    ) -> bool:
        for model_ref in (
            [config.base_model]
            + [e.source_model for e in config.experts]
            + [e.source_model for e in (config.shared_experts or [])]
        ):
            try:
                model_cfg = model_ref.config(trust_remote_code=trust_remote_code)
            except Exception:
                if explain:
                    logging.warning("Qwen3.5 MoE could not read a model config")
                return False
            if model_cfg.model_type not in SUPPORTED_MODEL_TYPES:
                if explain:
                    logging.warning(
                        "Qwen3.5 MoE only supports Qwen3.5 input models "
                        f"(got {model_cfg.model_type})"
                    )
                return False

        if len(config.shared_experts or []) > 1:
            if explain:
                logging.warning(
                    "Qwen3.5 MoE supports at most one shared expert; the dense "
                    "MLP is already kept as the always-on shared expert"
                )
            return False
        return True

    # ------------------------------------------------------------------ #
    # output config
    # ------------------------------------------------------------------ #
    def _generate_config(
        self,
        base_config: transformers.PretrainedConfig,
        num_experts: int,
        experts_per_token: Optional[int] = None,
    ) -> transformers.PretrainedConfig:
        data = base_config.to_dict()
        for key in VLM_ONLY_KEYS:
            data.pop(key, None)

        out_cfg = Qwen3_5MoeTextConfig(**data)
        out_cfg.architectures = ["Qwen3_5MoeForCausalLM"]

        # Belt and braces: some transformers versions keep the source model
        # type when it is passed through kwargs.
        if out_cfg.model_type != "qwen3_5_moe_text":
            out_cfg.model_type = "qwen3_5_moe_text"

        out_cfg.num_experts = max(int(num_experts), 1)
        out_cfg.num_experts_per_tok = max(
            1, min(experts_per_token or 2, out_cfg.num_experts)
        )
        out_cfg.moe_intermediate_size = out_cfg.intermediate_size
        # Dense MLP is preserved as the always-on shared expert.
        out_cfg.shared_expert_intermediate_size = out_cfg.intermediate_size
        out_cfg.norm_topk_prob = True

        if (out_cfg.num_experts & (out_cfg.num_experts - 1)) != 0:
            logging.warning(
                f"Your model has {out_cfg.num_experts} experts, which is not a "
                "power of two. The model will not be usable in llama.cpp."
            )
        return out_cfg

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    def write_model(
        self,
        out_path: str,
        config: MoEMergeConfig,
        merge_options: MergeOptions,
        router_weights: List[torch.Tensor],
        shared_router_weights: Optional[List[torch.Tensor]] = None,
    ):
        base_model = config.base_model
        base_cfg = base_model.config(trust_remote_code=merge_options.trust_remote_code)

        out_dtype = select_dtype(config, base_cfg)
        out_cfg = self._generate_config(
            base_cfg, len(config.experts), config.experts_per_token
        )
        if out_dtype is not None:
            out_cfg.torch_dtype = out_dtype
            out_cfg.dtype = out_dtype
        out_cfg.save_pretrained(out_path)

        loaders, base_loader, writer = initialize_io(config, out_path, merge_options)
        clone = merge_options.clone_tensors

        num_layers = int(getattr(base_cfg, "num_hidden_layers"))
        kinds = layer_kinds(base_cfg, num_layers)
        num_experts = len(config.experts)

        arch = arch_info_for_config(base_cfg)
        if arch is None:
            raise RuntimeError(
                f"No mergekit architecture is registered for "
                f"{base_cfg.architectures!r}. Add "
                f"mergekit/_data/architectures/{base_cfg.model_type}.json first."
            )

        # Every tensor the *text* model itself owns. Anything outside this set
        # is checkpoint-only data (mtp.*, vision towers, ...) carried over as-is.
        text_model_names = {w.name for w in arch.all_weights(base_cfg)}

        written: set = set()

        def save(name: str, tensor: torch.Tensor) -> None:
            writer.save_tensor(name, tensor.to(dtype=out_dtype), clone=clone)
            written.add(name)

        # ---- 1. non-MLP tensors straight from the base model -------------- #
        passthrough = [w for w in arch.all_weights(base_cfg) if not is_mlp_name(w.name)]
        for weight_info in tqdm.tqdm(passthrough, desc="Base weights"):
            tensor = find_tensor(
                base_loader, weight_info.name, weight_info.aliases or ()
            )
            if tensor is None:
                if weight_info.optional:
                    continue
                raise KeyError(weight_info.name)
            save(weight_info.name, tensor)

        # ---- 2. routed experts (fused layout) ---------------------------- #
        for layer_idx in tqdm.trange(num_layers, desc="Expert weights"):
            layer_prefix = f"model.layers.{layer_idx}"
            gate_up: List[torch.Tensor] = []
            down: List[torch.Tensor] = []
            for expert in config.experts:
                expert_loader = loaders.get(expert.source_model)
                params: Dict[str, torch.Tensor] = {}
                for param in ("gate_proj", "up_proj", "down_proj"):
                    tensor = find_tensor(
                        expert_loader, f"{layer_prefix}.mlp.{param}.weight"
                    )
                    if tensor is None:
                        raise KeyError(f"{layer_prefix}.mlp.{param}.weight")
                    params[param] = noise_and_scale(
                        tensor, expert, is_residual=param == "down_proj"
                    )
                gate_up.append(
                    torch.cat([params["gate_proj"], params["up_proj"]], dim=0)
                )
                down.append(params["down_proj"])

            save(
                f"{layer_prefix}.mlp.experts.gate_up_proj",
                torch.stack(gate_up, dim=0),
            )
            save(
                f"{layer_prefix}.mlp.experts.down_proj",
                torch.stack(down, dim=0),
            )

            # ---- 3. attention for exactly this layer's flavour ---------- #
            expected_kind = kinds[layer_idx]
            for weight_info in arch_layer_weights(arch, base_cfg, layer_idx):
                if is_mlp_name(weight_info.name):
                    continue
                if attention_kind(weight_info.name) != expected_kind:
                    continue
                tensor = find_tensor(
                    base_loader,
                    weight_info.name,
                    weight_info.aliases or (),
                )
                if tensor is None:
                    if weight_info.optional:
                        continue
                    raise KeyError(weight_info.name)
                save(weight_info.name, tensor)

        # ---- 4. router --------------------------------------------------- #
        for layer_idx, weight in enumerate(tqdm.tqdm(router_weights, desc="Router")):
            name = f"model.layers.{layer_idx}.mlp.gate.weight"
            writer.save_tensor(
                name,
                weight.to(dtype=out_dtype).contiguous(),
                clone=clone,
            )
            written.add(name)

        # ---- 5. shared expert = the layer's dense MLP --------------------- #
        shared_def: Optional[Expert] = (
            config.shared_experts[0] if config.shared_experts else None
        )
        shared_loader = (
            loaders.get(shared_def.source_model)
            if shared_def is not None
            else base_loader
        )
        for layer_idx in tqdm.trange(num_layers, desc="Shared expert"):
            layer_prefix = f"model.layers.{layer_idx}"
            for param in ("gate_proj", "up_proj", "down_proj"):
                tensor = find_tensor(
                    shared_loader, f"{layer_prefix}.mlp.{param}.weight"
                )
                if tensor is None:
                    raise KeyError(f"{layer_prefix}.mlp.{param}.weight")
                if shared_def is not None:
                    tensor = noise_and_scale(
                        tensor, shared_def, is_residual=param == "down_proj"
                    )
                save(f"{layer_prefix}.mlp.shared_expert.{param}.weight", tensor)

            # `shared_router_weights` is empty unless a shared expert with
            # prompts was configured; the gate must still be shape (1, hidden).
            gate = None
            if shared_router_weights and layer_idx < len(shared_router_weights):
                candidate = shared_router_weights[layer_idx]
                if candidate is not None and candidate.numel() == out_cfg.hidden_size:
                    gate = candidate.reshape(1, out_cfg.hidden_size)
            if gate is None:
                # Same default init as torch.nn.Linear: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
                scale = out_cfg.hidden_size**-0.5
                gate = (torch.rand(1, out_cfg.hidden_size) * 2 - 1) * scale
            name = f"{layer_prefix}.mlp.shared_expert_gate.weight"
            writer.save_tensor(name, gate.to(dtype=out_dtype).contiguous(), clone=clone)
            written.add(name)

        # ---- 6. checkpoint extras: mtp.*, vision towers, ... -------------- #
        extras = self._copy_extras(
            base_loader, writer, out_dtype, clone, text_model_names
        )

        writer.finalize()
        self._write_manifest(out_path, out_cfg, extras, kinds)
        logging.info(
            f"Qwen3.5 MoE written to {out_path}: {num_layers} layers, "
            f"{num_experts} routed experts, {len(extras)} extra tensors"
        )
        return out_path

    # ------------------------------------------------------------------ #
    @staticmethod
    def _copy_extras(
        base_loader,
        writer,
        out_dtype: Optional[torch.dtype],
        clone: bool,
        text_model_names: set,
    ) -> List[str]:
        """Carry over checkpoint tensors the text model does not own."""
        extras: List[str] = []
        available = set(base_loader.index.tensor_paths)
        for raw_name in sorted(available):
            name = strip_lm_prefix(raw_name)
            if name in text_model_names:
                # owned by the text model: rewritten above (or replaced by MoE)
                continue
            if name.startswith("model.layers.") and ".mlp." in name:
                # only *decoder* MLPs were replaced by the MoE block; MLPs that
                # belong to other sub-models (mtp.layers.*) must be carried over
                continue
            # load under whichever spelling the checkpoint actually uses
            source_name = next(
                (c for c in candidate_names(name) if c in available), raw_name
            )
            try:
                tensor = base_loader.get_tensor(source_name)
            except KeyError:
                continue
            if tensor is None:
                continue
            writer.save_tensor(name, tensor.to(dtype=out_dtype), clone=clone)
            extras.append(name)
        return extras

    @staticmethod
    def _write_manifest(
        out_path: str,
        out_cfg: transformers.PretrainedConfig,
        extras: Sequence[str],
        kinds: Sequence[str],
    ) -> None:
        manifest = {
            "output_architecture": "Qwen3.5 MoE",
            "model_type": out_cfg.model_type,
            "num_hidden_layers": int(getattr(out_cfg, "num_hidden_layers", 0)),
            "num_experts": int(getattr(out_cfg, "num_experts", 0)),
            "num_experts_per_tok": int(getattr(out_cfg, "num_experts_per_tok", 0)),
            "moe_intermediate_size": int(getattr(out_cfg, "moe_intermediate_size", 0)),
            "shared_expert_intermediate_size": int(
                getattr(out_cfg, "shared_expert_intermediate_size", 0)
            ),
            "layer_types": list(kinds),
            "extra_tensors": list(extras),
        }
        with open(
            os.path.join(out_path, "mergekit_moe.json"), "w", encoding="utf-8"
        ) as fp:
            json.dump(manifest, fp, indent=2)
