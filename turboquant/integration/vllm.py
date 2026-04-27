"""
TurboQuant vLLM integration — thin adapter layer.

Responsibilities:
  - Detect layer/backend type (flash vs MLA/GDN)
  - Install minimal monkey-patches that delegate to capture/store/score
  - Expose clean modes: off | capture_only | hybrid | full_tq
  - Keep patching surface tiny; all real logic lives in capture/store/score

Modes:
  - off:          no TQ activity, passthrough
  - capture_only: capture KV into compressed store, always use flash output
  - hybrid:       use compressed history + exact recent for decode
  - full_tq:      (future) TQ handles everything including prefill
"""

from __future__ import annotations

import math
import logging
import time
import types
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from turboquant.capture import KVCaptureEngine
from turboquant.store import CompressedKVStore
from turboquant.score import compute_hybrid_attention, MIN_HISTORY_FOR_TQ, preallocate_layer

logger = logging.getLogger("turboquant.integration.vllm")

MODE_OFF = "off"
MODE_CAPTURE_ONLY = "capture_only"
MODE_HYBRID = "hybrid"
MODE_FULL_TQ = "full_tq"
_VALID_MODES = (MODE_OFF, MODE_CAPTURE_ONLY, MODE_HYBRID, MODE_FULL_TQ)

_GLOBAL_MODE = MODE_CAPTURE_ONLY


def set_mode(mode: str):
    global _GLOBAL_MODE
    assert mode in _VALID_MODES, f"Invalid mode: {mode}. Valid: {_VALID_MODES}"
    _GLOBAL_MODE = mode
    logger.info(f"[TurboQuant] Mode set to: {mode}")


def get_mode() -> str:
    return _GLOBAL_MODE


@dataclass
class LayerConfig:
    """Per-layer TQ configuration."""
    head_dim: int
    num_kv_heads: int
    num_query_heads: int
    key_bits: int = 3
    value_bits: int = 2
    value_group_size: int = 32
    ring_capacity: int = 128
    layer_idx: int = 0
    backend_kind: str = "flash"  # "flash" | "mla"
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))


@dataclass
class LayerState:
    """Per-layer runtime state. Owns the capture engine and store."""
    config: LayerConfig
    store: CompressedKVStore
    engine: KVCaptureEngine
    _log_count: int = 0
    _no_alloc: bool = False

    @property
    def supports_hybrid(self) -> bool:
        return self.config.backend_kind == "flash"

    @property
    def graph_intended(self) -> bool:
        """True if CUDA Graph capture is expected (preallocated or no_alloc mode).

        When True, all torch.cuda.synchronize() and other graph-unsafe ops
        must be skipped.  Covers the case where preallocation failed but
        we're still in no_alloc mode (where vLLM will attempt graph capture).
        """
        return self.store.is_preallocated or self._no_alloc

    def reset(self):
        self.engine.reset()
        self._log_count = 0


def _create_layer_state(cfg: LayerConfig) -> LayerState:
    store = CompressedKVStore(
        head_dim=cfg.head_dim,
        num_kv_heads=cfg.num_kv_heads,
        key_bits=cfg.key_bits,
        value_bits=cfg.value_bits,
        value_group_size=cfg.value_group_size,
        device=cfg.device,
        layer_idx=cfg.layer_idx,
    )
    engine = KVCaptureEngine(
        store=store,
        ring_capacity=cfg.ring_capacity,
        device=cfg.device,
    )
    return LayerState(config=cfg, store=store, engine=engine)


def _infer_num_query_heads(attn_module, impl) -> int:
    for candidate in (
        getattr(attn_module, "num_heads", None),
        getattr(attn_module, "num_attention_heads", None),
        getattr(impl, "num_heads", None),
    ):
        if candidate:
            return int(candidate)
    return int(impl.num_kv_heads)


def _is_mla_impl(impl) -> bool:
    return (
        hasattr(impl, "forward_mqa")
        and hasattr(impl, "do_kv_cache_update")
        and not hasattr(impl, "forward")
    )


# ---------------------------------------------------------------------------
# Patched methods — kept as thin as possible
# ---------------------------------------------------------------------------

def _make_patched_kv_update(orig_fn, no_alloc: bool = False):
    """Intercept KV cache writes — skip original when no_alloc.

    KV capture is always done in patched forward to support KV-shared layers
    where vLLM skips calling do_kv_cache_update entirely.
    """

    def patched(self_impl, layer, key, value, kv_cache, slot_mapping):
        if not no_alloc:
            orig_fn(self_impl, layer, key, value, kv_cache, slot_mapping)
        # KV capture handled in patched forward (see capture_in_forward)

    return patched


def _no_alloc_prefill_attention(
    state: LayerState,
    self_impl,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata,
):
    num_actual = attn_metadata.num_actual_tokens
    q = query[:num_actual]
    k = key[:num_actual]
    v = value[:num_actual]

    if q.dim() == 2:
        q = q.view(num_actual, state.config.num_query_heads, state.config.head_dim)
    if k.dim() == 2:
        k = k.view(num_actual, state.config.num_kv_heads, state.config.head_dim)
        v = v.view(num_actual, state.config.num_kv_heads, state.config.head_dim)

    if state.config.num_query_heads != state.config.num_kv_heads:
        repeats = state.config.num_query_heads // state.config.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    q_t = q.unsqueeze(0).transpose(1, 2)
    k_t = k.unsqueeze(0).transpose(1, 2)
    v_t = v.unsqueeze(0).transpose(1, 2)

    scale = getattr(self_impl, "scale", 1.0 / math.sqrt(state.config.head_dim))
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
    return out.squeeze(0).transpose(0, 1)


def _make_patched_forward(orig_fn, state: LayerState, no_alloc: bool = False,
                          capture_in_forward: bool = False):
    """Intercept forward to optionally use TQ decode.

    If capture_in_forward=True, also capture K/V from forward args
    (needed when the backend has no separate do_kv_cache_update method).
    """

    # Diagnostic: track decode step count per-layer, only log first N
    _diag = {"step": 0}
    _DIAG_MAX_STEPS = 5

    def _capture_kv(key, value, attn_metadata, is_decode):
        """Capture K/V tensors into TQ store."""
        num_tokens = getattr(attn_metadata, 'num_actual_tokens', key.shape[0])
        if is_decode or num_tokens <= 1:
            # Decode (single or multi-sequence): ring buffer write is
            # graph-safe.  Multi-sequence decode (num_tokens > 1 but
            # is_decode=True) also goes through ingest_decode which
            # handles batched ring writes and overflow correctly.
            state.engine.ingest_decode(key[:num_tokens], value[:num_tokens], num_tokens)
        else:
            # Prefill (T>1): quantization is graph-safe (pre-allocated
            # tensors), so this can run during CUDA Graph capture too.
            # Warmup garbage is reset when actual inference starts
            # (ingest_prefill detects _was_decoding flag → reset()).
            state.engine.ingest_prefill(key[:num_tokens], value[:num_tokens], num_tokens)

    def patched(
        self_impl,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        mode = _GLOBAL_MODE
        is_decode = (attn_metadata is not None
                     and getattr(attn_metadata, 'max_query_len', 0) <= 1)
        # When store is pre-allocated or no_alloc mode is active, CUDA Graph
        # may be capturing at any time.  Disable all diagnostic timing
        # (torch.cuda.synchronize is forbidden during graph capture, and
        # is_current_stream_capturing() is unreliable when vLLM uses
        # non-default capture streams).
        _graph_intended = state.graph_intended
        should_log = is_decode and _diag["step"] < _DIAG_MAX_STEPS and not _graph_intended

        if should_log:
            torch.cuda.synchronize()
            t_total_0 = time.perf_counter()

        # Capture K/V when no separate kv_update hook exists.
        #
        # Decode (T=1): write_graph CUDA ops are the ONLY ops we want
        # recorded in the decode graph.  Always safe to call.
        #
        # Prefill (T>1) in EAGER mode (no CUDA Graph): safe to quantize
        # directly.
        #
        # Prefill (T>1) in GRAPH-INTENDED mode: vLLM's _dummy_run may
        # call the forward inside a torch.cuda.graph() context for
        # piecewise capture or profiling.  Running quantization here
        # would record CUDA ops in the graph that replay incorrectly
        # on every decode step.  Defer to the execute_model hook which
        # runs in eager mode between graph replays.
        if (capture_in_forward
                and mode not in (MODE_OFF,)
                and attn_metadata is not None):
            num_tok = getattr(attn_metadata, 'num_actual_tokens', key.shape[0])
            if is_decode or num_tok <= 1:
                _capture_kv(key, value, attn_metadata, is_decode)
            elif not _graph_intended:
                _capture_kv(key, value, attn_metadata, is_decode)
            else:
                # Graph-intended: defer prefill to execute_model hook
                state._pending_prefill_kv = (
                    key[:num_tok].clone(),
                    value[:num_tok].clone(),
                )
                state._need_prefill_reset = True

        if should_log:
            torch.cuda.synchronize()
            t_after_capture = time.perf_counter()

        # Off or capture-only: always use flash
        if mode in (MODE_OFF, MODE_CAPTURE_ONLY):
            ret = orig_fn(
                self_impl, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )
            if should_log:
                torch.cuda.synchronize()
                _diag["step"] += 1
                print(
                    f"[TQ-FWD] layer={state.config.layer_idx} "
                    f"path=orig_fn(mode={mode}) "
                    f"capture={(t_after_capture-t_total_0)*1000:.2f}ms "
                    f"total={(time.perf_counter()-t_total_0)*1000:.2f}ms",
                    flush=True,
                )
            return ret

        # Profiling pass or prefill: use flash
        if attn_metadata is None:
            return orig_fn(
                self_impl, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        is_prefill = attn_metadata.max_query_len > 1
        if is_prefill:
            if no_alloc:
                result = _no_alloc_prefill_attention(
                    state, self_impl, query, key, value, attn_metadata
                )
                num_actual = attn_metadata.num_actual_tokens
                result_flat = result.reshape(
                    num_actual, state.config.num_query_heads * state.config.head_dim
                ).to(query.dtype)
                if output is not None:
                    out_slice = output[:num_actual]
                    if out_slice.dim() == 3:
                        out_slice.copy_(result.to(out_slice.dtype))
                    else:
                        out_slice.copy_(result_flat.to(out_slice.dtype))
                    return output
                if query.dim() == 3:
                    return result.to(query.dtype)
                return result_flat
            return orig_fn(
                self_impl, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        # --- Hybrid decode ---
        if mode == MODE_HYBRID and state.supports_hybrid:
            if should_log:
                torch.cuda.synchronize()
                t_hybrid_start = time.perf_counter()

            num_actual = attn_metadata.num_actual_tokens
            q = query[:num_actual]
            if q.dim() == 2:
                q = q.view(num_actual, state.config.num_query_heads, state.config.head_dim)

            # Multi-token decode (T > 1) during CUDA Graph warmup: the graph-
            # compatible Triton kernel only supports T = 1.  Returning zeros is
            # safe because this only happens during memory profiling — actual
            # inference uses the T = 1 captured graph.
            if q.shape[0] > 1 and _graph_intended:
                # output is always provided by vLLM during graph capture.
                # Zero it out in-place (graph-safe).  Fall through to orig_fn
                # if somehow output is None (non-graph path won't hit this).
                if output is not None:
                    output[:num_actual].zero_()
                    return output
                return orig_fn(
                    self_impl, layer, query, key, value, kv_cache,
                    attn_metadata, output, output_scale, output_block_scale,
                )

            flat = state.store.get_flat_cache()

            if should_log:
                torch.cuda.synchronize()
                t_after_flat = time.perf_counter()

            if flat is not None and (state.store.is_preallocated or flat.num_tokens >= 16):

                # In CUDA-Graph mode, use full ring buffer (masking handled
                # in score.py's _hybrid_graph via device count tensor).
                # In eager mode, use Python-sliced peek().
                if state.engine.ring.graph_ready:
                    recent_k = state.engine.ring._k
                    recent_v = state.engine.ring._v
                else:
                    recent = state.engine.ring.peek()
                    recent_k = recent[0] if recent else None
                    recent_v = recent[1] if recent else None

                if should_log:
                    torch.cuda.synchronize()
                    t_after_prepare = time.perf_counter()

                result = compute_hybrid_attention(
                    query=q,
                    store=state.store,
                    recent_k=recent_k,
                    recent_v=recent_v,
                    num_query_heads=state.config.num_query_heads,
                    scale=getattr(self_impl, "scale", None),
                    layer_state=state,
                )

                if should_log:
                    torch.cuda.synchronize()
                    t_after_compute = time.perf_counter()

                result_flat = result.reshape(
                    num_actual, state.config.num_query_heads * state.config.head_dim
                ).to(query.dtype)

                if output is not None:
                    out_slice = output[:num_actual]
                    if out_slice.dim() == 3:
                        out_slice.copy_(result.to(out_slice.dtype))
                    else:
                        out_slice.copy_(result_flat.to(out_slice.dtype))
                    if should_log:
                        torch.cuda.synchronize()
                        _diag["step"] += 1
                        n_hist = flat.num_tokens
                        n_recent = recent_k.shape[0] if recent_k is not None else 0
                        print(
                            f"[TQ-FWD] layer={state.config.layer_idx} "
                            f"path=hybrid hist={n_hist} recent={n_recent} "
                            f"capture={(t_after_capture-t_total_0)*1000:.2f}ms "
                            f"get_flat={(t_after_flat-t_hybrid_start)*1000:.2f}ms "
                            f"prepare={(t_after_prepare-t_after_flat)*1000:.2f}ms "
                            f"compute={(t_after_compute-t_after_prepare)*1000:.2f}ms "
                            f"total={(time.perf_counter()-t_total_0)*1000:.2f}ms",
                            flush=True,
                        )
                    return output
                if query.dim() == 3:
                    return result.to(query.dtype)
                return result_flat

        # Fallback to flash
        if no_alloc:
            # no_alloc mode: paged cache is not populated, so flash attention
            # would read garbage. Use TQ attention (compressed + exact recent).
            num_actual = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
            q_fb = query[:num_actual]
            if q_fb.dim() == 2:
                q_fb = q_fb.view(
                    num_actual,
                    state.config.num_query_heads,
                    state.config.head_dim,
                )

            # Same T > 1 shortcut as above (CUDA Graph warmup profiling).
            if q_fb.shape[0] > 1 and _graph_intended:
                if output is not None:
                    output[:num_actual].zero_()
                    return output
                return orig_fn(
                    self_impl, layer, query, key, value, kv_cache,
                    attn_metadata, output, output_scale, output_block_scale,
                )

            flat = state.store.get_flat_cache()
            has_history = flat is not None and (
                state.store.is_preallocated or flat.num_tokens >= MIN_HISTORY_FOR_TQ
            )

            # In graph mode, ring buffer is always non-empty after write_graph
            if state.engine.ring.graph_ready:
                has_recent = True
                recent_k = state.engine.ring._k
                recent_v = state.engine.ring._v
            else:
                recent = state.engine.ring.peek()
                has_recent = recent is not None and recent[0].shape[0] > 0
                recent_k = recent[0] if recent else None
                recent_v = recent[1] if recent else None

            if has_history or has_recent:
                result = compute_hybrid_attention(
                    query=q_fb,
                    store=state.store,
                    recent_k=recent_k,
                    recent_v=recent_v,
                    num_query_heads=state.config.num_query_heads,
                    scale=getattr(self_impl, "scale", None),
                    layer_state=state,
                )

                result_flat = result.reshape(
                    num_actual, state.config.num_query_heads * state.config.head_dim
                ).to(query.dtype)

                if output is not None:
                    out_slice = output[:num_actual]
                    if out_slice.dim() == 3:
                        out_slice.copy_(result.to(out_slice.dtype))
                    else:
                        out_slice.copy_(result_flat.to(out_slice.dtype))
                    if should_log:
                        torch.cuda.synchronize()
                        _diag["step"] += 1
                        n_hist = flat.num_tokens if flat else 0
                        n_recent = recent_k.shape[0] if recent_k is not None else 0
                        print(
                            f"[TQ-FWD] layer={state.config.layer_idx} "
                            f"path=no_alloc_fallback hist={n_hist} recent={n_recent} "
                            f"total={(time.perf_counter()-t_total_0)*1000:.2f}ms",
                            flush=True,
                        )
                    return output
                if query.dim() == 3:
                    return result.to(query.dtype)
                return result_flat

            # No KV data at all (very first decode before ring is populated).
            # During graph capture, must not allocate new tensors.
            if should_log:
                _diag["step"] += 1
                print(
                    f"[TQ-FWD] layer={state.config.layer_idx} "
                    f"path=zeros(no_data) "
                    f"total={(time.perf_counter()-t_total_0)*1000:.2f}ms",
                    flush=True,
                )
            num_actual = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
            if output is not None:
                output[:num_actual].zero_()
                return output
            # Fallback: use orig_fn (never reached during graph capture)
            return orig_fn(
                self_impl, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        if should_log:
            torch.cuda.synchronize()
            _diag["step"] += 1
            print(
                f"[TQ-FWD] layer={state.config.layer_idx} "
                f"path=orig_fn(fallback) "
                f"total={(time.perf_counter()-t_total_0)*1000:.2f}ms",
                flush=True,
            )
        return orig_fn(
            self_impl, layer, query, key, value, kv_cache,
            attn_metadata, output, output_scale, output_block_scale,
        )

    return patched


def _make_patched_mla_update(orig_fn, state: LayerState):
    """MLA KV update — log-only, no TQ capture yet."""

    def patched(self_impl, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale):
        orig_fn(self_impl, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale)
        if state._log_count < 1:
            logger.info(
                f"[TurboQuant] MLA update observed on layer {state.config.layer_idx}; "
                "TQ MLA path is deferred."
            )
            state._log_count += 1

    return patched


def _make_patched_mla_forward(orig_fn, state: LayerState):
    """MLA forward — passthrough (unsupported)."""

    def patched(self_impl, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        return orig_fn(self_impl, q, kv_c_and_k_pe_cache, attn_metadata, layer)

    return patched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def install_hooks(
    model_runner,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    ring_capacity: int = 128,
    initial_layers_count: int = 4,
    initial_layers_key_bits: int | None = None,
    mode: str = MODE_CAPTURE_ONLY,
    no_alloc: bool = False,
) -> dict[str, LayerState]:
    """Install TurboQuant hooks on all attention layers in a vLLM model runner.

    Returns: dict mapping layer_name -> LayerState
    """
    global _GLOBAL_MODE
    _GLOBAL_MODE = mode

    if initial_layers_key_bits is None:
        initial_layers_key_bits = min(key_bits + 1, 4)

    static_ctx = model_runner.compilation_config.static_forward_context
    device = model_runner.device

    layer_states: dict[str, LayerState] = {}
    layer_idx = 0

    for layer_name, attn_module in static_ctx.items():
        if not hasattr(attn_module, "impl"):
            continue

        impl = attn_module.impl
        num_kv_heads = getattr(impl, "num_kv_heads", None)
        if num_kv_heads is None:
            continue

        if hasattr(impl, "head_size"):
            head_dim = int(impl.head_size)
        elif hasattr(impl, "kv_lora_rank"):
            head_dim = int(impl.kv_lora_rank)
        else:
            continue

        bits = initial_layers_key_bits if layer_idx < initial_layers_count else key_bits
        backend_kind = "mla" if _is_mla_impl(impl) else "flash"
        num_query_heads = _infer_num_query_heads(attn_module, impl)

        cfg = LayerConfig(
            head_dim=head_dim,
            num_kv_heads=int(num_kv_heads),
            num_query_heads=num_query_heads,
            key_bits=bits,
            value_bits=value_bits,
            value_group_size=min(value_group_size, head_dim),
            ring_capacity=ring_capacity,
            layer_idx=layer_idx,
            backend_kind=backend_kind,
            device=device,
        )

        state = _create_layer_state(cfg)
        state._no_alloc = no_alloc
        layer_states[layer_name] = state

        if backend_kind == "flash":
            has_separate_kv_update = hasattr(impl, "do_kv_cache_update")

            if has_separate_kv_update:
                # Patch to skip original paged cache write (no_alloc),
                # but do NOT capture KV here — always done in forward
                # to support KV-shared layers where vLLM skips this call.
                patched_update = _make_patched_kv_update(
                    impl.do_kv_cache_update.__func__, no_alloc=no_alloc
                )
                impl.do_kv_cache_update = types.MethodType(
                    lambda self, *a, _p=patched_update, **kw: _p(self, *a, **kw), impl
                )

            # Always capture KV in forward. For KV-shared layers,
            # do_kv_cache_update is never called by vLLM, so forward
            # is the only place to capture.
            patched_forward = _make_patched_forward(
                impl.forward.__func__, state, no_alloc=no_alloc,
                capture_in_forward=True,
            )
            impl.forward = types.MethodType(
                lambda self, *a, _p=patched_forward, **kw: _p(self, *a, **kw), impl
            )

        else:
            if hasattr(impl, "do_kv_cache_update"):
                patched_update = _make_patched_mla_update(impl.do_kv_cache_update.__func__, state)
                impl.do_kv_cache_update = types.MethodType(
                    lambda self, *a, _p=patched_update, **kw: _p(self, *a, **kw), impl
                )
            if hasattr(impl, "forward_mqa"):
                patched_fwd = _make_patched_mla_forward(impl.forward_mqa.__func__, state)
                impl.forward_mqa = types.MethodType(
                    lambda self, *a, _p=patched_fwd, **kw: _p(self, *a, **kw), impl
                )

        impl._tq_layer_state = state
        layer_idx += 1

    model_runner._tq_layer_states = layer_states
    model_runner._tq_no_alloc = no_alloc
    logger.info(
        f"[TurboQuant] Hooks on {len(layer_states)} layers "
        f"(mode={mode}, no_alloc={no_alloc})"
    )

    # Pre-allocate CUDA-Graph-compatible buffers.
    # Derive max_tokens from the model's scheduling config.
    max_tokens = 0
    try:
        scheduler_config = getattr(model_runner, 'scheduler_config', None)
        if scheduler_config is not None:
            max_tokens = getattr(scheduler_config, 'max_num_seqs', 0) * getattr(
                scheduler_config, 'max_model_len', 0
            )
            # Use max_model_len as a generous upper bound per sequence
            max_tokens = getattr(scheduler_config, 'max_model_len', 0)
    except Exception:
        pass

    # Fallback: try model_config if scheduler_config didn't work
    if max_tokens <= 0:
        try:
            model_config = getattr(model_runner, 'model_config', None)
            if model_config is not None:
                max_tokens = getattr(model_config, 'max_model_len', 0)
        except Exception:
            pass

    print(
        f"[TQ-PREALLOC] max_tokens={max_tokens} no_alloc={no_alloc} "
        f"n_hybrid={sum(1 for s in layer_states.values() if s.supports_hybrid)}",
        flush=True,
    )

    if max_tokens > 0:
        for name, state in layer_states.items():
            if state.supports_hybrid:
                try:
                    preallocate_layer(state, max_tokens)
                except Exception as e:
                    logger.warning(
                        "[TurboQuant] Pre-allocation failed for %s: %s", name, e
                    )
    else:
        if no_alloc:
            logger.error(
                "[TurboQuant] CRITICAL: no_alloc=True but max_tokens=%d — "
                "preallocation skipped! CUDA Graph capture WILL fail. "
                "Check scheduler_config availability.",
                max_tokens,
            )

    # Log preallocation results
    n_prealloc = sum(
        1 for s in layer_states.values()
        if s.supports_hybrid and s.store.is_preallocated
    )
    n_graph_ready = sum(
        1 for s in layer_states.values()
        if s.supports_hybrid and s.engine.ring._graph_mode
    )
    print(
        f"[TQ-PREALLOC] result: stores_ready={n_prealloc} "
        f"ring_ready={n_graph_ready} graph_intended={no_alloc}",
        flush=True,
    )

    # Patch model runner to detect ring buffer overflow and compress
    # between CUDA Graph replays.  This Python hook runs after every
    # execute_model call; the overhead is negligible (CPU counter check).
    _tq_states_ref = layer_states
    _orig_execute_model = getattr(model_runner, 'execute_model', None)
    if _orig_execute_model is not None and not getattr(model_runner, '_tq_execute_patched', False):
        def _make_tq_execute_hook(orig_fn, states):
            # Track prefill-to-decode transitions to distinguish
            # warmup (skip) from real inference (process deferred KV).
            _transitions_seen = 0

            def hooked(*args, **kwargs):
                nonlocal _transitions_seen
                result = orig_fn(*args, **kwargs)
                for _name, st in states.items():
                    ring = st.engine.ring
                    if ring._graph_mode:
                        ring._cpu_decode_steps += 1

                        # Handle prefill-reset signal from patched forward.
                        if getattr(st, '_need_prefill_reset', False):
                            st.engine._was_decoding = False
                            st._need_prefill_reset = False

                        if not st.engine._was_decoding:
                            # Prefill-to-decode transition.
                            st.engine._was_decoding = True
                            _transitions_seen += 1

                            # Process deferred prefill KV.  Skip the FIRST
                            # transition (during warmup/capture) — the data
                            # is fake and the store will be reset on the
                            # next real prefill anyway.
                            if (_transitions_seen >= 2
                                    and getattr(st, '_pending_prefill_kv', None) is not None):
                                k, v = st._pending_prefill_kv
                                st.engine.ingest_prefill(k, v, k.shape[0])
                                st._pending_prefill_kv = None

                        st.engine.check_overflow_and_compress()
                return result
            return hooked

        model_runner.execute_model = _make_tq_execute_hook(
            _orig_execute_model, _tq_states_ref,
        )
        model_runner._tq_execute_patched = True
        logger.info("[TurboQuant] Patched model_runner.execute_model for overflow check")

    # Diagnostic init log
    n_flash = sum(1 for s in layer_states.values() if s.config.backend_kind == "flash")
    n_mla = len(layer_states) - n_flash
    print(
        f"[TQ-INIT] layers={len(layer_states)} flash={n_flash} mla={n_mla} "
        f"mode={mode} no_alloc={no_alloc} "
        f"capture_in_forward=True compute_path=score.py_pytorch",
        flush=True,
    )
    for name, s in layer_states.items():
        print(
            f"[TQ-INIT]   {name}: backend={s.config.backend_kind} "
            f"kv_heads={s.config.num_kv_heads} q_heads={s.config.num_query_heads} "
            f"head_dim={s.config.head_dim} key_bits={s.config.key_bits} "
            f"value_bits={s.config.value_bits} "
            f"prealloc={s.store.is_preallocated} graph={s.graph_intended}",
            flush=True,
        )

    return layer_states


def free_kv_cache(model_runner) -> int:
    """Free paged KV cache for TQ-hooked layers. Returns bytes freed.

    Only frees layers that have TQ state. Non-TQ layers (MLA/GDN) keep their cache.
    """
    layer_states = getattr(model_runner, "_tq_layer_states", None)
    if not layer_states:
        logger.warning("[TurboQuant] No layer states found, nothing to free")
        return 0

    static_ctx = model_runner.compilation_config.static_forward_context
    device = model_runner.device
    freed = 0
    tiny = torch.zeros(1, dtype=torch.int8, device=device)

    ptrs_to_free = set()
    for layer_name, state in layer_states.items():
        if not state.supports_hybrid:
            continue
        if layer_name not in static_ctx:
            continue
        attn_module = static_ctx[layer_name]
        kv_list = getattr(attn_module, "kv_cache", None)
        if kv_list and len(kv_list) > 0:
            ptrs_to_free.add(kv_list[0].data_ptr())

    for layer_name, state in layer_states.items():
        if not state.supports_hybrid:
            continue
        if layer_name not in static_ctx:
            continue
        attn_module = static_ctx[layer_name]
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
    logger.info(f"[TurboQuant] Freed {freed / 1e6:.0f} MB KV cache ({len(layer_states)} layers)")
    return freed


def get_stats(model_runner) -> dict:
    """Return summary statistics for all TQ layer states."""
    layer_states = getattr(model_runner, "_tq_layer_states", None)
    if not layer_states:
        return {}

    stats = {}
    total_compressed = 0
    total_buffered = 0
    total_memory = 0

    for name, state in layer_states.items():
        compressed = state.store.num_tokens
        buffered = state.engine.ring.size
        mem = state.store.memory_bytes()
        total_compressed += compressed
        total_buffered += buffered
        total_memory += mem

    stats["num_layers"] = len(layer_states)
    stats["total_compressed_tokens"] = total_compressed // max(len(layer_states), 1)
    stats["total_buffered_tokens"] = total_buffered // max(len(layer_states), 1)
    stats["total_memory_bytes"] = total_memory
    stats["mode"] = _GLOBAL_MODE
    return stats
