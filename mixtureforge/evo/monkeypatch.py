# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only


import transformers


def monkeypatch_lmeval_shuffle():
    """Monkeypatch lm_eval to shuffle the dataset after downloading."""
    import lm_eval.api.task

    if hasattr(lm_eval.api.task.Task, "_monkey_patched"):
        return

    _old_task_dl = lm_eval.api.task.Task.download

    def _dl_shuffled(self: lm_eval.api.task.Task, *args, **kwargs):
        _old_task_dl(self, *args, **kwargs)
        self.dataset = self.dataset.shuffle()

    lm_eval.api.task.Task.download = _dl_shuffled

    _old_ct_dl = lm_eval.api.task.ConfigurableTask.download

    def _ct_dl_shuffled(self, *args, **kwargs):
        _old_ct_dl(self, *args, **kwargs)
        self.dataset = self.dataset.shuffle()

    lm_eval.api.task.ConfigurableTask.download = _ct_dl_shuffled

    lm_eval.api.task.Task._monkey_patched = True
    print("monkey has been patched")


def monkeypatch_tqdm(lm_eval: bool = True, mixtureforge: bool = True):
    """Patch lm_eval & mixtureforge to use Ray's tqdm for progress bars."""

    from ray.experimental.tqdm_ray import tqdm as tqdm_ray

    def _tqdm_wrap(iterable=None, disable: bool = False, **kwargs):
        if disable:
            if iterable is not None:
                return iterable
            return lambda x: x
        res = tqdm_ray(iterable=iterable, **kwargs, flush_interval_s=1.0)
        res.refresh()
        return res

    def _patch_lm_eval():
        import lm_eval

        if hasattr(lm_eval, "_mk_tqdm_patched"):
            return

        import lm_eval.api.metrics
        import lm_eval.api.model
        import lm_eval.api.task
        import lm_eval.models.huggingface
        import lm_eval.models.vllm_causallms

        for module in (
            lm_eval.models.huggingface,
            lm_eval.models.vllm_causallms,
            lm_eval.api.model,
            lm_eval.api.task,
            lm_eval.api.metrics,
        ):
            setattr(module, "tqdm", _tqdm_wrap)

        lm_eval._mk_tqdm_patched = True

    if lm_eval:
        _patch_lm_eval()

    if mixtureforge:
        del mixtureforge

        import mixtureforge
        import mixtureforge.graph
        import mixtureforge.merge
        import mixtureforge.tokenizer

        fake_module = type("fake_module", (), {"tqdm": staticmethod(_tqdm_wrap)})()

        mixtureforge.graph.tqdm = fake_module
        mixtureforge.merge.tqdm = fake_module
        mixtureforge.tokenizer.tqdm = fake_module


def monkeypatch_lmeval_vllm():
    # HACK: fix crash on some tasks due to unset AUTO_MODEL_CLASS for vLLM
    import lm_eval.models.vllm_causallms

    lm_eval.models.vllm_causallms.VLLM.AUTO_MODEL_CLASS = (
        transformers.AutoModelForCausalLM
    )
