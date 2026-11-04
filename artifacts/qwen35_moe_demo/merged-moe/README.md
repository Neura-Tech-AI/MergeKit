---
base_model: /tmp/clone/dense-a
tags:
- mergekit
- merge
- mixture-of-experts
library_name: transformers
---

# out-moe

This is a Mixture-of-Experts model built with [mergekit](https://github.com/arcee-ai/mergekit)
from pre-trained dense language models, using the **Qwen3.5 MoE** output architecture.

## Merge Details

Each expert below is a dense model converted into a set of routed experts. The
router was generated with `gate_mode: hidden` and routes each token to
`2` expert(s).

| Role | Model | Prompts |
|---|---|---|
| expert 0 | /tmp/clone/dense-a | write python code function parse json error handle, debug stack trace explain root cause |
| expert 1 | /tmp/clone/dense-b | prove square is irrational step solve problem carefully verify answer, think logic puzzle before answering |
| expert 2 | /tmp/clone/dense-c | story short lighthouse keeper describe scene vivid poetic language, continue narrative engaging style |
| shared 0 | /tmp/clone/dense-a | always active |

## Configuration

The following YAML configuration was used to produce this model:

```yaml
base_model: /tmp/clone/dense-a

experts:
  - source_model: /tmp/clone/dense-a
    positive_prompts:
      - "write python code function parse json error handle"
      - "debug stack trace explain root cause"
  - source_model: /tmp/clone/dense-b
    positive_prompts:
      - "prove square is irrational step solve problem carefully verify answer"
      - "think logic puzzle before answering"
  - source_model: /tmp/clone/dense-c
    positive_prompts:
      - "story short lighthouse keeper describe scene vivid poetic language"
      - "continue narrative engaging style"

shared_experts:
  - source_model: /tmp/clone/dense-a

gate_mode: hidden
experts_per_token: 2
dtype: bfloat16

```
