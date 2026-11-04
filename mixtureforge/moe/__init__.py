from typing import List

from mixtureforge.moe.arch import MoEOutputArchitecture
from mixtureforge.moe.deepseek import DeepseekMoE
from mixtureforge.moe.mixtral import MixtralMoE

ALL_OUTPUT_ARCHITECTURES: List[MoEOutputArchitecture] = [MixtralMoE(), DeepseekMoE()]

try:
    from mixtureforge.moe.qwen import QwenMoE
except ImportError:
    pass
else:
    ALL_OUTPUT_ARCHITECTURES.append(QwenMoE())

try:
    from mixtureforge.moe.qwen3 import Qwen3MoE
except ImportError:
    pass
else:
    ALL_OUTPUT_ARCHITECTURES.append(Qwen3MoE())

try:
    from mixtureforge.moe.qwen3_5 import Qwen3_5MoE
except ImportError:
    pass
else:
    ALL_OUTPUT_ARCHITECTURES.append(Qwen3_5MoE())

__all__ = [
    "ALL_OUTPUT_ARCHITECTURES",
    "MoEOutputArchitecture",
]
