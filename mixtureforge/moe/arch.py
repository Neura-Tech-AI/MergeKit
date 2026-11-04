# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from mixtureforge.moe.config import MoEMergeConfig
from mixtureforge.options import MergeOptions


class MoEOutputArchitecture(ABC):
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for the architecture."""
        pass

    @abstractmethod
    def supports_config(
        self,
        config: MoEMergeConfig,
        explain: bool = False,
        trust_remote_code: bool = False,
    ) -> bool:
        """Return whether this architecture supports the given config.

        If `explain` is True, log an explanation of why the config is not supported."""
        pass

    def tagalong_files(self) -> List[str]:
        """Extra non-weight files this output architecture needs copied.

        These complement the files mixtureforge copies by default (tokenizer,
        special tokens, chat template, generation config). Architectures that
        need preprocessors or other side files override this.
        """
        return []

    @abstractmethod
    def write_model(
        self,
        out_path: str,
        config: MoEMergeConfig,
        merge_options: MergeOptions,
        router_weights: List[torch.Tensor],
        shared_router_weights: Optional[List[torch.Tensor]] = None,
    ):
        """Write the config and tensors for the output MoE to the given path."""
        pass
