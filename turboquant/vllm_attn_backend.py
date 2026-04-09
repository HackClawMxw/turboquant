"""
TurboQuant attention backend shim for vLLM v0.17+.

Delegates to turboquant.integration.vllm for all real logic.
Kept for backward compatibility with scripts that import from here.
"""

from __future__ import annotations

import logging

import torch

import turboquant.integration.vllm as _new_backend

logger = logging.getLogger("turboquant.attn")

MODE_SHADOW = "shadow"
MODE_ACCUMULATE = "accumulate"
MODE_ACTIVE = "active"
_VALID_MODES = (MODE_SHADOW, MODE_ACCUMULATE, MODE_ACTIVE)

_LEGACY_TO_NEW = {
    MODE_ACCUMULATE: _new_backend.MODE_CAPTURE_ONLY,
    MODE_SHADOW: _new_backend.MODE_CAPTURE_ONLY,
    MODE_ACTIVE: _new_backend.MODE_HYBRID,
}

_GLOBAL_MODE = MODE_ACCUMULATE


def set_mode(mode: str):
    global _GLOBAL_MODE
    assert mode in _VALID_MODES
    _GLOBAL_MODE = mode
    _new_backend.set_mode(_LEGACY_TO_NEW.get(mode, _new_backend.MODE_CAPTURE_ONLY))


def get_mode() -> str:
    return _GLOBAL_MODE


def install_turboquant_hooks(
    model_runner,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    buffer_size: int = 128,
    initial_layers_count: int = 4,
    initial_layers_key_bits: int | None = None,
    mode: str = MODE_ACCUMULATE,
    no_alloc: bool = False,
):
    global _GLOBAL_MODE
    new_mode = _LEGACY_TO_NEW.get(mode, _new_backend.MODE_CAPTURE_ONLY)

    layer_states = _new_backend.install_hooks(
        model_runner,
        key_bits=key_bits,
        value_bits=value_bits,
        value_group_size=value_group_size,
        ring_capacity=buffer_size,
        initial_layers_count=initial_layers_count,
        initial_layers_key_bits=initial_layers_key_bits,
        mode=new_mode,
        no_alloc=no_alloc,
    )

    _GLOBAL_MODE = mode
    model_runner._tq_states = layer_states
    model_runner._tq_no_alloc = no_alloc
    return layer_states


_TQ_NO_ALLOC_CONFIG = None


def enable_no_alloc(
    key_bits: int = 3,
    value_bits: int = 2,
    buffer_size: int = 128,
    initial_layers_count: int = 4,
):
    """Call BEFORE creating vllm.LLM(). Patches the executor so TQ hooks
    are installed automatically during engine initialization."""
    global _TQ_NO_ALLOC_CONFIG
    _TQ_NO_ALLOC_CONFIG = dict(
        key_bits=key_bits,
        value_bits=value_bits,
        buffer_size=buffer_size,
        initial_layers_count=initial_layers_count,
    )

    from vllm.v1.executor.abstract import Executor

    if hasattr(Executor, "_tq_patched"):
        return

    orig_get_specs = Executor.get_kv_cache_specs

    def patched_get_kv_cache_specs(self):
        cfg = _TQ_NO_ALLOC_CONFIG
        logger.debug("patched_get_kv_cache_specs called, cfg=%s", cfg is not None)
        if cfg is None:
            return orig_get_specs(self)

        def _worker_install_tq(worker):
            from turboquant.vllm_attn_backend import (
                install_turboquant_hooks, MODE_ACTIVE
            )
            tq_states = install_turboquant_hooks(
                worker.model_runner,
                key_bits=cfg["key_bits"],
                value_bits=cfg["value_bits"],
                buffer_size=cfg["buffer_size"],
                initial_layers_count=cfg["initial_layers_count"],
                mode=MODE_ACTIVE,
                no_alloc=True,
            )

            # NOTE: Do NOT set kv_sharing_target_layer_name here.
            # In vLLM v0.18.0+, Attention.forward() skips calling
            # do_kv_cache_update when kv_sharing_target_layer_name is set.
            # Since no_alloc mode relies on patched do_kv_cache_update to
            # capture KV into the TQ store, sharing would prevent KV capture
            # for all shared layers, causing garbage output.

            flash_layers = sum(
                1 for s in tq_states.values()
                if getattr(s, "supports_hybrid", False)
            )
            return {
                "hooks": len(tq_states),
                "flash_layers": flash_layers,
            }

        try:
            hooks = self.collective_rpc(_worker_install_tq)
            logger.info("[TurboQuant] Installed no_alloc hooks: %s", hooks)
        except Exception as e:
            logger.error("[TurboQuant] collective_rpc FAILED: %s", e, exc_info=True)
        return orig_get_specs(self)

    Executor.get_kv_cache_specs = patched_get_kv_cache_specs
    Executor._tq_patched = True

    logger.info("[TurboQuant] Patched Executor for auto TQ hook installation")


def free_kv_cache(model_runner):
    """Free paged KV cache for TQ-hooked layers."""
    if getattr(model_runner, "_tq_layer_states", None):
        return _new_backend.free_kv_cache(model_runner)

    layer_states = getattr(model_runner, "_tq_states", None)
    if not layer_states:
        return 0

    static_ctx = model_runner.compilation_config.static_forward_context
    device = model_runner.device
    freed = 0
    tiny = torch.zeros(1, dtype=torch.int8, device=device)

    ptrs_to_free = set()
    for layer_name, state in layer_states.items():
        if not getattr(state, "supports_hybrid", False):
            continue
        attn_module = static_ctx.get(layer_name)
        if attn_module is None:
            continue
        kv_list = getattr(attn_module, "kv_cache", None)
        if kv_list and len(kv_list) > 0 and hasattr(kv_list[0], "data_ptr"):
            ptrs_to_free.add(kv_list[0].data_ptr())

    for layer_name, state in layer_states.items():
        if not getattr(state, "supports_hybrid", False):
            continue
        attn_module = static_ctx.get(layer_name)
        if attn_module is None:
            continue
        kv_list = getattr(attn_module, "kv_cache", None)
        if kv_list and len(kv_list) > 0:
            old = kv_list[0]
            freed += old.nelement() * old.element_size()
            kv_list[0] = tiny

    for i in range(len(model_runner.kv_caches)):
        entry = model_runner.kv_caches[i]
        if isinstance(entry, list):
            for j in range(len(entry)):
                if hasattr(entry[j], "data_ptr") and entry[j].data_ptr() in ptrs_to_free:
                    entry[j] = tiny
        elif hasattr(entry, "data_ptr") and entry.data_ptr() in ptrs_to_free:
            model_runner.kv_caches[i] = tiny

    torch.cuda.empty_cache()
    return freed
