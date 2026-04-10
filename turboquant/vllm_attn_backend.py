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

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        GPUModelRunner = None

    if hasattr(Executor, "_tq_patched"):
        return

    # Patch layout update so shared layers point to the target layer's cache.
    # Needed for hybrid attention+mamba models where shared layers don't get
    # their own cache allocated (specs removed).
    if (
        GPUModelRunner is not None
        and not hasattr(GPUModelRunner, "_tq_layout_patch")
        and hasattr(GPUModelRunner, "_update_hybrid_attention_mamba_layout")
    ):
        _orig_layout_update = GPUModelRunner._update_hybrid_attention_mamba_layout

        def _patched_layout_update(self_runner, kv_caches):
            for layer_name, target_name in getattr(
                self_runner, "shared_kv_cache_layers", {}
            ).items():
                if layer_name not in kv_caches and target_name in kv_caches:
                    kv_caches[layer_name] = kv_caches[target_name]
            return _orig_layout_update(self_runner, kv_caches)

        GPUModelRunner._update_hybrid_attention_mamba_layout = _patched_layout_update
        GPUModelRunner._tq_layout_patch = True

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

            # Set up KV sharing so all flash layers share the first layer's
            # paged cache.  This reduces KV cache allocation from N to 1.
            # KV capture is done in patched forward (capture_in_forward=True),
            # so skipping do_kv_cache_update for shared layers is safe.
            static_ctx = worker.model_runner.compilation_config.static_forward_context
            flash_layers = [
                name for name, state in tq_states.items()
                if getattr(state, "supports_hybrid", False)
            ]
            shared_layer_names = []
            if len(flash_layers) > 1:
                target = flash_layers[0]
                target_attn = static_ctx.get(target)
                if target_attn is not None and hasattr(
                    target_attn, "kv_sharing_target_layer_name"
                ):
                    target_attn.kv_sharing_target_layer_name = None
                for name in flash_layers[1:]:
                    attn = static_ctx.get(name)
                    if attn is not None and hasattr(
                        attn, "kv_sharing_target_layer_name"
                    ):
                        attn.kv_sharing_target_layer_name = target
                        shared_layer_names.append(name)

            return {
                "hooks": len(tq_states),
                "flash_layers": len(flash_layers),
                "shared_layer_names": shared_layer_names,
            }

        try:
            hooks = self.collective_rpc(_worker_install_tq)
            logger.info("[TurboQuant] Installed no_alloc hooks: %s", hooks)
        except Exception as e:
            logger.error("[TurboQuant] collective_rpc FAILED: %s", e, exc_info=True)
            return orig_get_specs(self)

        specs = orig_get_specs(self)

        # Remove specs for shared layers so vLLM doesn't allocate KV cache
        # for them — they'll share the target layer's cache instead.
        shared = []
        if hooks and isinstance(hooks, list) and len(hooks) > 0:
            shared = hooks[0].get("shared_layer_names", [])
        for name in shared:
            specs.pop(name, None)
        if shared:
            logger.info(
                "[TurboQuant] Removed %d shared layer specs from KV cache "
                "allocation (layers share one paged cache)", len(shared),
            )

        return specs

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
