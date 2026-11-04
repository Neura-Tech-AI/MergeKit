# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import logging
import math
from typing import Dict, List, Union

import torch
import tqdm
import transformers
from transformers import AutoModelForCausalLM, LlamaForCausalLM, MistralForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from mergekit.common import ModelReference
from mergekit.moe.config import Expert


def get_hidden_states(
    model: Union[MistralForCausalLM, LlamaForCausalLM],
    tokenized: transformers.BatchEncoding,
    average: bool = True,
) -> List[torch.Tensor]:
    with torch.no_grad():
        output: CausalLMOutputWithPast = model(
            **tokenized.to(model.device), output_hidden_states=True, return_dict=True
        )
    hidden_states = torch.stack(
        output.hidden_states[:-1]
    )  # (num_layers, batch_size, seq_len, hidden_size)
    if average:
        # use average over each prompt's tokens, leaving out the batch padding
        mask = tokenized["attention_mask"].to(hidden_states)[None, :, :, None]
        hidden_states = (hidden_states * mask).sum(dim=2) / mask.sum(dim=2)
    else:
        # take last value
        hidden_states = hidden_states[:, :, -1, :]
    return hidden_states.sum(dim=1) / hidden_states.shape[1]


def get_router_input_states(
    model: transformers.PreTrainedModel,
    tokenized: transformers.BatchEncoding,
    average: bool = True,
) -> torch.Tensor:
    """Gate vectors taken from the exact tensor the router sees at inference.

    `get_hidden_states` reads `output_hidden_states[i]`, the *pre-norm residual
    stream*: what enters decoder layer i, not what enters its MoE block. The
    router is fed `hidden + attn(post_attention_layernorm(hidden))`, i.e. the
    post-attention normalised state. Building gate vectors from the residual
    stream puts them in a different space than the queries they are scored
    against, which shows up as near-random routing on smaller models.

    This hooks every decoder MLP / MoE block and records its input, which is
    exactly the router's query.

    Returns `(num_layers, hidden_size)`.
    """
    layers = None
    for attr in ("model", "language_model.model", "transformer"):
        candidate = getattr(model, attr, None)
        if candidate is not None and hasattr(candidate, "layers"):
            layers = candidate.layers
            break
    if layers is None:
        raise RuntimeError("Unable to locate decoder layers for gate generation")

    captured: Dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_module, inputs, _output):
            value = inputs[0] if inputs else None
            if torch.is_tensor(value) and value.dim() >= 2:
                captured[layer_idx] = value.detach()

        return hook

    targets = []
    for layer_idx, layer in enumerate(layers):
        module = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if module is not None:
            targets.append((layer_idx, module))

    handles = [
        module.register_forward_hook(make_hook(layer_idx))
        for layer_idx, module in targets
    ]
    try:
        with torch.no_grad():
            model(**tokenized.to(model.device), return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    if not captured:
        raise RuntimeError("No decoder MLP inputs were captured")

    rows = []
    mask = tokenized.get("attention_mask")
    for layer_idx in sorted(captured):
        value = captured[layer_idx]
        if value.dim() == 2:  # already flattened by the block
            rows.append(value[-1])
            continue
        if average and mask is not None:
            m = mask.to(value)[:, :, None]
            rows.append(((value * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)).mean(0))
        else:
            rows.append(value[:, -1, :].mean(0))
    return torch.stack(rows, dim=0)  # (num_layers, hidden_size)


def get_cheap_embedding(
    embed: torch.Tensor,
    tokenized: Dict[str, torch.Tensor],
    num_layers: int,
    vocab_size: int,
) -> torch.Tensor:
    onehot = torch.nn.functional.one_hot(
        tokenized["input_ids"], num_classes=vocab_size
    )  # (batch_size, seq_len, 32000)
    h = onehot.float() @ embed.float()  # (batch_size, seq_len, hidden_size)
    embedded = (
        (h * tokenized["attention_mask"].unsqueeze(-1))
        .sum(dim=1)
        .sum(dim=0, keepdim=True)
    )  # (1, hidden_size)
    res = embedded / embedded.norm(dim=-1, keepdim=True).clamp(
        min=1e-8
    )  # (1, hidden_size)
    return res.repeat(num_layers, 1)


def tokenize_prompts(
    prompts: List[str], tokenizer: transformers.PreTrainedTokenizerBase
):
    return tokenizer(
        [(tokenizer.bos_token or "") + p for p in prompts],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )


def get_gate_params(
    model_ref: ModelReference,
    tokenizer: transformers.PreTrainedTokenizerBase,
    experts: List[Expert],
    mode: str = "hidden",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    lazy_unpickle: bool = False,
    trust_remote_code: bool = False,
    device: str = "auto",
):
    gate_vecs = []
    _do_it = None

    model_cfg = model_ref.config(trust_remote_code=trust_remote_code)

    if mode == "random":
        return torch.randn(
            (model_cfg.num_hidden_layers, len(experts), model_cfg.hidden_size)
        )
    elif mode == "uniform_random":
        in_features = model_cfg.hidden_size
        scale = math.sqrt(1.0 / in_features)
        return (
            torch.rand(
                (model_cfg.num_hidden_layers, len(experts), model_cfg.hidden_size)
            )
            * 2
            * scale
            - scale
        )
    elif mode == "cheap_embed":
        embed = model_ref.lazy_loader(lazy_unpickle=lazy_unpickle).get_tensor(
            "model.embed_tokens.weight"
        )

        def _do_it(tokenized):
            return get_cheap_embedding(
                embed,
                tokenized,
                num_layers=model_cfg.num_hidden_layers,
                vocab_size=model_cfg.vocab_size,
            )

    elif mode in ("hidden", "hidden_avg", "hidden_last", "router_input"):
        kwargs = {}
        if load_in_4bit:
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True
            )
        elif load_in_8bit:
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_8bit=True
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_ref.model.path,
            revision=model_ref.model.revision,
            torch_dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        if mode == "router_input":
            captured_average = True

            def _do_it(tokenized):
                vecs = get_router_input_states(
                    model, tokenized=tokenized, average=captured_average
                )
                return vecs.to(torch.float32)

        else:

            def _do_it(tokenized):
                return get_hidden_states(
                    model, tokenized=tokenized, average=mode == "hidden_avg"
                )

    gate_vecs = []
    for expert in tqdm.tqdm(experts, desc="expert prompts"):
        hidden_states = _do_it(tokenize_prompts(expert.positive_prompts, tokenizer))
        if expert.negative_prompts:
            hidden_states -= _do_it(
                tokenize_prompts(expert.negative_prompts, tokenizer)
            )

        hidden_states /= hidden_states.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        gate_vecs.append(hidden_states)
    gate_vecs = torch.stack(gate_vecs, dim=0)  # (num_expert, num_layer, hidden_size)
    return gate_vecs.permute(1, 0, 2)


def warn_degenerate_gates(gate_vecs: torch.Tensor, threshold: float = 5.0):
    degen_indices = []
    num_layers, _num_experts, _hidden_size = gate_vecs.shape
    for idx in range(num_layers):
        c = torch.linalg.cond(gate_vecs[idx, :, :].float())
        if c > threshold:
            degen_indices.append(idx)

    if degen_indices:
        if len(degen_indices) == 1:
            layer_str = f"layer {degen_indices[0]}"
            verb = "has"
        elif len(degen_indices) == 2:
            layer_str = f"layers {' and '.join(map(str, degen_indices))}"
            verb = "have"
        elif len(degen_indices) >= num_layers:
            layer_str = "ALL layers"
            verb = "have"
        else:
            layer_str = (
                "layers "
                + ", ".join(map(str, degen_indices[:-1]))
                + ", and "
                + str(degen_indices[-1])
            )
            verb = "have"

        logging.warning(
            f"{layer_str} {verb} degenerate routing parameters "
            "- your prompts may be too similar."
        )
        logging.warning("One or more experts will be underutilized in your model.")
