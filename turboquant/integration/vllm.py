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


class LayerSlotPool:
    """Per-request slot pool for a single attention layer.

    Manages ``max_num_seqs`` independent LayerState instances so that
    concurrent requests don't share KV state.

    Only slot 0 is created at startup.  Extra slots are created lazily
    in ``allocate()`` when first needed, to avoid OOM from creating
    all quantizers (rotation matrices) upfront.
    """

    def __init__(
        self,
        slot0: LayerState,
        cfg: LayerConfig,
        max_slots: int = 1,
        no_alloc: bool = False,
    ):
        self.slots: list[LayerState] = [slot0]
        self._block_to_slot: dict[int, int] = {}  # first_block_idx -> slot_idx
        self._free = [0]
        self._max_slots = max_slots
        self._cfg = cfg
        self._no_alloc = no_alloc
        self._lazy_prealloc_tokens: int = 0  # set by install_hooks

    def _create_slot(self) -> int:
        """Create a new LayerState on-demand and return its index."""
        st = _create_layer_state(self._cfg)
        st._no_alloc = self._no_alloc
        idx = len(self.slots)
        self.slots.append(st)
        return idx

    def allocate(self, first_block: int) -> LayerState:
        """Allocate a slot for a new request. Returns LayerState."""
        if first_block in self._block_to_slot:
            slot_idx = self._block_to_slot[first_block]
            # Block reused from a finished request — reset stale data.
            self.slots[slot_idx].reset()
            return self.slots[slot_idx]

        # Need a free slot — create one lazily if allowed.
        if not self._free:
            if len(self.slots) < self._max_slots:
                idx = self._create_slot()
                self._free.append(idx)
            else:
                # All slots occupied — evict the oldest (first-inserted) entry.
                oldest_block = next(iter(self._block_to_slot))
                evict_idx = self._block_to_slot.pop(oldest_block)
                self.slots[evict_idx].reset()
                self._free.append(evict_idx)

        slot_idx = self._free.pop(0)
        self._block_to_slot[first_block] = slot_idx
        # Lazy pre-allocate buffers for this slot on first use.
        # Slot 0 is pre-allocated in install_hooks; extra slots are
        # pre-allocated here to avoid OOM at startup.
        st = self.slots[slot_idx]
        if st.supports_hybrid and not st.store.is_preallocated and self._lazy_prealloc_tokens > 0:
            from turboquant.score import preallocate_layer
            try:
                preallocate_layer(st, self._lazy_prealloc_tokens)
            except Exception as e:
                logger.warning(
                    "[TurboQuant] Lazy preallocation failed for slot %d: %s", slot_idx, e,
                )
        return st

    def get(self, first_block: int) -> LayerState:
        """Get the slot for an existing request."""
        return self.slots[self._block_to_slot[first_block]]

    def release(self, first_block: int):
        """Release a slot when a request finishes."""
        if first_block in self._block_to_slot:
            slot_idx = self._block_to_slot.pop(first_block)
            self.slots[slot_idx].reset()
            self._free.append(slot_idx)

    def active_slots(self) -> list[LayerState]:
        """Return all currently active slots."""
        return [self.slots[i] for i in self._block_to_slot.values()]


def _get_block_table(attn_metadata) -> Optional[torch.Tensor]:
    """Extract block_table from attn_metadata across vLLM versions.

    Returns:
        (num_reqs, max_blocks) int tensor, or None if unavailable.
    """
    bt = getattr(attn_metadata, 'block_table', None)
    if bt is None:
        bt = getattr(attn_metadata, 'block_table_tensor', None)
    if bt is None:
        for meta_name in ('decode_metadata', 'prefill_metadata'):
            meta = getattr(attn_metadata, meta_name, None)
            if meta is not None:
                bt = getattr(meta, 'block_tables', None)
                if bt is not None:
                    break
    return bt


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


def _make_patched_forward(orig_fn, pool_or_state, no_alloc: bool = False,
                          capture_in_forward: bool = False):
    """Intercept forward to optionally use TQ decode.

    Args:
        pool_or_state: LayerSlotPool (multi-slot) or LayerState (single-slot / MLA).

    If capture_in_forward=True, also capture K/V from forward args
    (needed when the backend has no separate do_kv_cache_update method).
    """

    # Accept both pool (new) and single state (MLA fallback)
    if isinstance(pool_or_state, LayerSlotPool):
        _pool = pool_or_state
        _single_state = None
    else:
        _pool = None
        _single_state = pool_or_state

    # Diagnostic: track decode step count per-layer, only log first N
    _diag = {"step": 0}
    _DIAG_MAX_STEPS = 5

    def _capture_kv_for_slot(st, key, value, attn_metadata, is_decode):
        """Capture K/V tensors into a specific slot's TQ store."""
        num_tokens = getattr(attn_metadata, 'num_actual_tokens', key.shape[0])
        if is_decode or num_tokens <= 1:
            st.engine.ingest_decode(key[:num_tokens], value[:num_tokens], num_tokens)
        else:
            st.engine.ingest_prefill(key[:num_tokens], value[:num_tokens], num_tokens)

    def _resolve_slot(attn_metadata, is_prefill):
        """Resolve the LayerState for the current request.

        For single-state mode (MLA), returns _single_state.
        For pool mode, uses block_table to find/allocate the slot.
        """
        if _pool is None:
            return _single_state

        block_table = _get_block_table(attn_metadata)
        if block_table is None:
            # No block table available — fall back to slot 0
            return _pool.slots[0]

        if is_prefill:
            # Prefill: allocate a new slot for this request
            first_block = int(block_table[0, 0])
            return _pool.allocate(first_block)
        else:
            # Decode: look up existing slot.
            # For multi-sequence decode, return slot for first sequence.
            # Multi-seq loop handles per-seq slot resolution below.
            first_block = int(block_table[0, 0])
            try:
                return _pool.get(first_block)
            except KeyError:
                # Slot not found (shouldn't happen in normal flow)
                return _pool.allocate(first_block)

    def _resolve_slot_for_seq(attn_metadata, seq_idx):
        """Resolve the LayerState for a specific decode sequence.

        In CUDA Graph mode, the graph binds to tensor addresses during
        warmup.  Position i must map to slot i so that replay uses the
        correct per-slot tensors.  When block_table is unavailable
        (warmup / graph capture), we assign by position index directly.
        """
        block_table = _get_block_table(attn_metadata)
        if block_table is None:
            # Warmup or missing block table: assign by position index.
            # This is critical for CUDA Graph — position i → slot i ensures
            # the graph captures per-slot operations for each decode token.
            return _pool.slots[min(seq_idx, len(_pool.slots) - 1)]
        first_block = int(block_table[seq_idx, 0])
        try:
            return _pool.get(first_block)
        except KeyError:
            return _pool.allocate(first_block)

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
        is_prefill = (attn_metadata is not None
                      and getattr(attn_metadata, 'max_query_len', 0) > 1)

        # Resolve the primary slot for this request.
        if _pool is not None:
            state = _resolve_slot(attn_metadata, is_prefill)
        else:
            state = _single_state

        _graph_intended = state.graph_intended
        should_log = is_decode and _diag["step"] < _DIAG_MAX_STEPS and not _graph_intended

        if should_log:
            torch.cuda.synchronize()
            t_total_0 = time.perf_counter()

        # --- KV Capture ---
        if (capture_in_forward
                and mode not in (MODE_OFF,)
                and attn_metadata is not None):
            num_tok = getattr(attn_metadata, 'num_actual_tokens', key.shape[0])

            # Per-request KV capture using query_start_loc.
            # Handles pure prefill, pure decode, and mixed batches correctly.
            # query_start_loc: (num_reqs+1,) cumulative query token counts.
            _qsl = getattr(attn_metadata, 'query_start_loc', None)
            _nr = getattr(attn_metadata, 'num_reqs', None)
            # Derive num_reqs from query_start_loc when not in metadata.
            if _nr is None and _qsl is not None and len(_qsl) > 1:
                _nr = len(_qsl) - 1

            if _pool is not None and _qsl is not None and _nr is not None and _nr > 0:
                for si in range(_nr):
                    start = int(_qsl[si])
                    end = int(_qsl[si + 1])
                    qlen = end - start
                    st = _resolve_slot_for_seq(attn_metadata, si)
                    if qlen > 1:
                        st.engine.ingest_prefill(
                            key[start:end], value[start:end], qlen
                        )
                    elif qlen == 1:
                        st.engine.ingest_decode(
                            key[start:end], value[start:end], 1
                        )
            elif is_decode or num_tok <= 1:
                # Fallback (no query_start_loc): pure decode path
                if _pool is not None and num_tok > 1:
                    for si in range(num_tok):
                        st = _resolve_slot_for_seq(attn_metadata, si)
                        st.engine.ingest_decode(
                            key[si:si+1], value[si:si+1], 1
                        )
                else:
                    state.engine.ingest_decode(key[:num_tok], value[:num_tok], num_tok)
            elif not no_alloc:
                # Eager mode (no_alloc=False): capture prefill directly.
                # NOTE: do NOT use _graph_intended here — it is True whenever
                # the store is preallocated, even in eager mode.  Using it
                # would defer the prefill to the execute_model hook, but the
                # hook only processes deferred prefill when ring._graph_mode
                # is True (which it isn't in eager mode), so the prefill
                # would be silently dropped.
                _capture_kv_for_slot(state, key, value, attn_metadata, False)
            else:
                # Graph-intended: defer prefill to execute_model hook
                state._pending_prefill_kv = (
                    key[:num_tok].clone(),
                    value[:num_tok].clone(),
                )
                state._need_prefill_reset = True

            # Diagnostic: log key state transitions on layer 0 only.
            # Print on: prefill, nr change, or first 2 multi-req decode steps.
            if state.config.layer_idx == 0:
                _prev_nr = _diag.get("prev_nr", 0)
                _do_print = False
                if is_prefill:
                    _do_print = True
                elif _nr is not None and _nr != _prev_nr:
                    _do_print = True
                elif is_decode and _nr is not None and _nr > 1 and _diag.get("multi_dec", 0) < 2:
                    _do_print = True
                    _diag["multi_dec"] = _diag.get("multi_dec", 0) + 1
                if _do_print:
                    _bt = _get_block_table(attn_metadata)
                    _path = ("per_req_qsl" if (_qsl is not None and _nr is not None)
                             else "decode" if is_decode
                             else "direct_prefill" if not no_alloc
                             else "deferred_prefill")
                    _slot_info = ""
                    if _pool is not None and _bt is not None:
                        _slot_info = f" b2s={dict(_pool._block_to_slot)}"
                    print(
                        f"[TQ-CAPTURE] layer={state.config.layer_idx} "
                        f"path={_path} is_decode={is_decode} is_prefill={is_prefill} "
                        f"num_tok={num_tok} qsl={_qsl is not None} nr={_nr} "
                        f"no_alloc={no_alloc} pool={_pool is not None}{_slot_info}",
                        flush=True,
                    )
                    _diag["prev_nr"] = _nr

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

            # Multi-token decode (T > 1) during CUDA Graph warmup.
            # Only trigger when actually using CUDA Graph (no_alloc=True).
            # _graph_intended is True whenever the store is pre-allocated,
            # which includes eager mode — using it here would incorrectly
            # zero out multi-sequence decode output in eager mode.
            if q.shape[0] > 1 and no_alloc:
                if output is not None:
                    output[:num_actual].zero_()
                    return output
                return orig_fn(
                    self_impl, layer, query, key, value, kv_cache,
                    attn_metadata, output, output_scale, output_block_scale,
                )

            # Multi-sequence decode: compute per-sequence attention.
            # In CUDA Graph mode this only runs during single-token replay
            # (T=1), so num_reqs will be > 1 only in eager mode.
            num_reqs = getattr(attn_metadata, 'num_reqs', None)
            if num_reqs is None:
                # Fallback: in pure decode each request contributes 1 token
                num_reqs = q.shape[0]
            if _pool is not None and num_reqs > 1:
                for si in range(num_reqs):
                    st = _resolve_slot_for_seq(attn_metadata, si)
                    q_i = q[si:si+1]
                    flat = st.store.get_flat_cache()

                    if flat is not None and (st.store.is_preallocated or flat.num_tokens >= 16):
                        if st.engine.ring.graph_ready:
                            recent_k = st.engine.ring._k
                            recent_v = st.engine.ring._v
                        else:
                            recent = st.engine.ring.peek()
                            recent_k = recent[0] if recent else None
                            recent_v = recent[1] if recent else None

                        result_i = compute_hybrid_attention(
                            query=q_i,
                            store=st.store,
                            recent_k=recent_k,
                            recent_v=recent_v,
                            num_query_heads=st.config.num_query_heads,
                            scale=getattr(self_impl, "scale", None),
                            layer_state=st,
                        )
                    else:
                        # No compressed history yet: use exact recent only
                        if st.engine.ring.graph_ready:
                            recent_k = st.engine.ring._k
                            recent_v = st.engine.ring._v
                        else:
                            recent = st.engine.ring.peek()
                            recent_k = recent[0] if recent else None
                            recent_v = recent[1] if recent else None

                        if recent_k is not None:
                            result_i = compute_hybrid_attention(
                                query=q_i,
                                store=st.store,
                                recent_k=recent_k,
                                recent_v=recent_v,
                                num_query_heads=st.config.num_query_heads,
                                scale=getattr(self_impl, "scale", None),
                                layer_state=st,
                            )
                        else:
                            # No data at all — zeros
                            result_i = torch.zeros(
                                1, st.config.num_query_heads, st.config.head_dim,
                                device=q_i.device, dtype=q_i.dtype,
                            )

                    result_flat_i = result_i.reshape(
                        1, state.config.num_query_heads * state.config.head_dim
                    )
                    if output is not None:
                        out_slice_i = output[si:si+1]
                        if out_slice_i.dim() == 3:
                            out_slice_i.copy_(result_i)
                        else:
                            out_slice_i.copy_(result_flat_i)
                    # If no output tensor, we can't write per-seq results
                    # without allocation — this shouldn't happen in vLLM decode

                if should_log:
                    _diag["step"] += 1
                    print(
                        f"[TQ-FWD] layer={state.config.layer_idx} "
                        f"path=hybrid_multi_seq seqs={num_reqs} "
                        f"total={(time.perf_counter()-t_total_0)*1000:.2f}ms",
                        flush=True,
                    )
                if output is not None:
                    return output

            # Single-sequence decode (or no pool)
            flat = state.store.get_flat_cache()

            if should_log:
                torch.cuda.synchronize()
                t_after_flat = time.perf_counter()

            if flat is not None and (state.store.is_preallocated or flat.num_tokens >= 16):

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
                )

                if output is not None:
                    out_slice = output[:num_actual]
                    if out_slice.dim() == 3:
                        out_slice.copy_(result)
                    else:
                        out_slice.copy_(result_flat)
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
                    return result
                return result_flat

        # Fallback to flash
        if no_alloc:
            num_actual = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
            q_fb = query[:num_actual]
            if q_fb.dim() == 2:
                q_fb = q_fb.view(
                    num_actual,
                    state.config.num_query_heads,
                    state.config.head_dim,
                )

            # Multi-token decode during CUDA Graph warmup.
            # Only trigger when actually using CUDA Graph (no_alloc=True).
            if q_fb.shape[0] > 1 and no_alloc:
                if output is not None:
                    output[:num_actual].zero_()
                    return output
                return orig_fn(
                    self_impl, layer, query, key, value, kv_cache,
                    attn_metadata, output, output_scale, output_block_scale,
                )

            # Multi-sequence decode in no_alloc fallback.
            # In CUDA Graph mode this only runs during single-token replay.
            num_reqs = getattr(attn_metadata, 'num_reqs', None)
            if num_reqs is None:
                num_reqs = q_fb.shape[0]
            if _pool is not None and num_reqs > 1:
                for si in range(num_reqs):
                    st = _resolve_slot_for_seq(attn_metadata, si)
                    q_i = q_fb[si:si+1]
                    flat = st.store.get_flat_cache()
                    has_history = flat is not None and (
                        st.store.is_preallocated or flat.num_tokens >= MIN_HISTORY_FOR_TQ
                    )

                    if st.engine.ring.graph_ready:
                        has_recent = True
                        recent_k = st.engine.ring._k
                        recent_v = st.engine.ring._v
                    else:
                        recent = st.engine.ring.peek()
                        has_recent = recent is not None and recent[0].shape[0] > 0
                        recent_k = recent[0] if recent else None
                        recent_v = recent[1] if recent else None

                    if has_history or has_recent:
                        result_i = compute_hybrid_attention(
                            query=q_i,
                            store=st.store,
                            recent_k=recent_k,
                            recent_v=recent_v,
                            num_query_heads=st.config.num_query_heads,
                            scale=getattr(self_impl, "scale", None),
                            layer_state=st,
                        )
                        result_flat_i = result_i.reshape(
                            1, st.config.num_query_heads * st.config.head_dim
                        )
                        if output is not None:
                            out_slice_i = output[si:si+1]
                            if out_slice_i.dim() == 3:
                                out_slice_i.copy_(result_i)
                            else:
                                out_slice_i.copy_(result_flat_i)

                if output is not None:
                    if should_log:
                        _diag["step"] += 1
                        print(
                            f"[TQ-FWD] layer={state.config.layer_idx} "
                            f"path=no_alloc_multi_seq seqs={num_reqs} "
                            f"total={(time.perf_counter()-t_total_0)*1000:.2f}ms",
                            flush=True,
                        )
                    return output

            # Single-sequence no_alloc fallback
            flat = state.store.get_flat_cache()
            has_history = flat is not None and (
                state.store.is_preallocated or flat.num_tokens >= MIN_HISTORY_FOR_TQ
            )

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
                )

                if output is not None:
                    out_slice = output[:num_actual]
                    if out_slice.dim() == 3:
                        out_slice.copy_(result)
                    else:
                        out_slice.copy_(result_flat)
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
                    return result
                return result_flat

            # No KV data at all (very first decode before ring is populated).
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
    max_num_seqs: int = 1,
) -> dict[str, LayerSlotPool]:
    """Install TurboQuant hooks on all attention layers in a vLLM model runner.

    Args:
        max_num_seqs: Number of concurrent request slots to preallocate per layer.

    Returns: dict mapping layer_name -> LayerSlotPool
    """
    global _GLOBAL_MODE
    _GLOBAL_MODE = mode

    if initial_layers_key_bits is None:
        initial_layers_key_bits = min(key_bits + 1, 4)

    static_ctx = model_runner.compilation_config.static_forward_context
    device = model_runner.device

    layer_pools: dict[str, LayerSlotPool] = {}
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

        # Only create slot 0 upfront.  Extra slots are created lazily
        # in LayerSlotPool.allocate() when first needed.  This avoids
        # OOM at startup — each extra slot's quantizer (rotation matrices,
        # codebooks, etc.) is only allocated when a concurrent request
        # actually needs it.
        pool = LayerSlotPool(
            slot0=state, cfg=cfg, max_slots=max_num_seqs, no_alloc=no_alloc,
        )
        layer_pools[layer_name] = pool

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
                impl.forward.__func__, pool, no_alloc=no_alloc,
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

        impl._tq_layer_pool = pool
        layer_idx += 1

    model_runner._tq_layer_pools = layer_pools
    # Keep backward compat alias for code that reads _tq_layer_states
    model_runner._tq_layer_states = {
        name: pool.slots[0] for name, pool in layer_pools.items()
    }
    model_runner._tq_no_alloc = no_alloc
    logger.info(
        f"[TurboQuant] Hooks on {len(layer_pools)} layers "
        f"(mode={mode}, no_alloc={no_alloc}, slots={max_num_seqs})"
    )

    if no_alloc and max_num_seqs > 1:
        logger.info(
            "[TurboQuant] no_alloc=True with max_num_seqs=%d: "
            "CUDA Graph does NOT support per-request KV isolation. "
            "Use enforce_eager=True for correct multi-request support.",
            max_num_seqs,
        )

    if max_num_seqs < 2:
        logger.warning(
            "[TurboQuant] max_num_seqs=%d: only 1 slot per layer. "
            "Concurrent requests will evict each other's KV data. "
            "Set max_num_seqs >= expected concurrent requests "
            "(or use default 0 for auto-detect from vLLM scheduler).",
            max_num_seqs,
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
        f"n_hybrid_pools={sum(1 for p in layer_pools.values() if p.slots[0].supports_hybrid)} "
        f"slots_per_pool={max_num_seqs}",
        flush=True,
    )

    if max_tokens > 0:
        for name, pool in layer_pools.items():
            # Store max_tokens for lazy preallocation of extra slots.
            pool._lazy_prealloc_tokens = max_tokens
            # Only pre-allocate slot 0 at startup to avoid OOM.
            # Extra slots are lazily pre-allocated in LayerSlotPool.allocate()
            # when they are first assigned to a request.
            slot0 = pool.slots[0]
            if slot0.supports_hybrid:
                try:
                    preallocate_layer(slot0, max_tokens)
                except Exception as e:
                    logger.warning(
                        "[TurboQuant] Pre-allocation failed for %s slot 0: %s", name, e
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
    all_slots = [s for p in layer_pools.values() for s in p.slots]
    n_prealloc = sum(
        1 for s in all_slots
        if s.supports_hybrid and s.store.is_preallocated
    )
    n_graph_ready = sum(
        1 for s in all_slots
        if s.supports_hybrid and s.engine.ring._graph_mode
    )
    print(
        f"[TQ-PREALLOC] result: stores_ready={n_prealloc} "
        f"ring_ready={n_graph_ready} graph_intended={no_alloc} "
        f"total_slots={len(all_slots)}",
        flush=True,
    )

    # Patch model runner to detect ring buffer overflow and compress
    # between CUDA Graph replays.  This Python hook runs after every
    # execute_model call; the overhead is negligible (CPU counter check).
    _tq_pools_ref = layer_pools
    _orig_execute_model = getattr(model_runner, 'execute_model', None)
    if _orig_execute_model is not None and not getattr(model_runner, '_tq_execute_patched', False):
        def _make_tq_execute_hook(orig_fn, pools):
            _hook_step = {"n": 0}
            def hooked(*args, **kwargs):
                result = orig_fn(*args, **kwargs)
                for _name, pool in pools.items():
                    _b2s = dict(pool._block_to_slot)
                    for st in pool.active_slots():
                        ring = st.engine.ring

                        # Process deferred prefill regardless of graph mode.
                        # In eager mode, _graph_intended may be True (because
                        # the store is preallocated), causing patched forward
                        # to defer prefill.  Without this, the deferred prefill
                        # would be silently dropped in eager mode.
                        if getattr(st, '_need_prefill_reset', False):
                            st.engine._was_decoding = False
                            st._need_prefill_reset = False

                        if not st.engine._was_decoding:
                            st.engine._was_decoding = True

                            if getattr(st, '_pending_prefill_kv', None) is not None:
                                k, v = st._pending_prefill_kv
                                st.engine.ingest_prefill(k, v, k.shape[0])
                                st._pending_prefill_kv = None

                                if _hook_step["n"] < 5:
                                    print(
                                        f"[TQ-HOOK] layer={st.config.layer_idx} "
                                        f"processed deferred prefill tok={k.shape[0]} "
                                        f"graph_mode={ring._graph_mode} "
                                        f"pos={ring._pos} b2s={_b2s}",
                                        flush=True,
                                    )

                                if ring._graph_mode and ring._pos >= ring.capacity:
                                    ring._pos = 0
                                    ring._pos_tensor.fill_(0)
                                    ring._count_tensor.fill_(ring.capacity)
                                if ring._graph_mode:
                                    ring._cpu_decode_steps = 0

                        if ring._graph_mode:
                            ring._cpu_decode_steps += 1
                            st.engine.check_overflow_and_compress()
                _hook_step["n"] += 1
                return result
            return hooked

        model_runner.execute_model = _make_tq_execute_hook(
            _orig_execute_model, _tq_pools_ref,
        )
        model_runner._tq_execute_patched = True
        logger.info("[TurboQuant] Patched model_runner.execute_model for overflow check")

    # Diagnostic init log
    # Count distinct layers (slot 0 per pool), not total slots
    n_flash = sum(1 for p in layer_pools.values() if p.slots[0].config.backend_kind == "flash")
    n_mla = len(layer_pools) - n_flash
    print(
        f"[TQ-INIT] layers={len(layer_pools)} flash={n_flash} mla={n_mla} "
        f"mode={mode} no_alloc={no_alloc} slots={max_num_seqs} "
        f"capture_in_forward=True compute_path=score.py_pytorch",
        flush=True,
    )
    for name, pool in layer_pools.items():
        s = pool.slots[0]
        print(
            f"[TQ-INIT]   {name}: backend={s.config.backend_kind} "
            f"kv_heads={s.config.num_kv_heads} q_heads={s.config.num_query_heads} "
            f"head_dim={s.config.head_dim} key_bits={s.config.key_bits} "
            f"value_bits={s.config.value_bits} slots={len(pool.slots)} "
            f"prealloc={s.store.is_preallocated} graph={s.graph_intended}",
            flush=True,
        )

    return layer_pools


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
    layer_pools = getattr(model_runner, "_tq_layer_pools", None)
    if not layer_pools:
        # Fallback to old-style _tq_layer_states
        layer_states = getattr(model_runner, "_tq_layer_states", None)
        if not layer_states:
            return {}
        stats = {}
        total_compressed = 0
        total_buffered = 0
        total_memory = 0
        for name, state in layer_states.items():
            total_compressed += state.store.num_tokens
            total_buffered += state.engine.ring.size
            total_memory += state.store.memory_bytes()
        stats["num_layers"] = len(layer_states)
        stats["total_compressed_tokens"] = total_compressed // max(len(layer_states), 1)
        stats["total_buffered_tokens"] = total_buffered // max(len(layer_states), 1)
        stats["total_memory_bytes"] = total_memory
        stats["mode"] = _GLOBAL_MODE
        return stats

    stats = {}
    total_compressed = 0
    total_buffered = 0
    total_memory = 0
    active_count = 0

    for name, pool in layer_pools.items():
        for slot in pool.active_slots():
            total_compressed += slot.store.num_tokens
            total_buffered += slot.engine.ring.size
            total_memory += slot.store.memory_bytes()
            active_count += 1

    stats["num_layers"] = len(layer_pools)
    stats["active_slots"] = active_count
    stats["total_compressed_tokens"] = total_compressed // max(active_count, 1)
    stats["total_buffered_tokens"] = total_buffered // max(active_count, 1)
    stats["total_memory_bytes"] = total_memory
    stats["mode"] = _GLOBAL_MODE
    return stats
