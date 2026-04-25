"""
TurboQuant score module — CUDA-Graph-compatible decode attention.

When the store is pre-allocated (``store.is_preallocated``), the fused
kernel reads N from a device tensor via a while loop — no dynamic
tensor allocation, no Python-dependent shapes.  Fully compatible with
CUDA Graph capture/replay.

Fallback paths (score-only Triton, pure PyTorch) remain for non-graph
use or when Triton is unavailable.
"""

from __future__ import annotations

import math
import logging
import time
import torch
import torch.nn.functional as F
from typing import Optional

from turboquant.store import FlatCache, CompressedKVStore
from turboquant.kv_cache import dequantize_values
from turboquant.quantizer import TurboQuantProd

logger = logging.getLogger("turboquant.score")

MIN_HISTORY_FOR_TQ = 16

# ---- Diagnostic timing ----
_DIAG_MAX_LOG = 10
_diag_log_count = 0

# ---- Triton availability ----
_FUSED_AVAILABLE = False
_SCORE_AVAILABLE = False
try:
    from turboquant.triton_kernels import turboquant_fused_decode_gqa
    _FUSED_AVAILABLE = True
    _SCORE_AVAILABLE = True
except (ImportError, Exception) as _e:
    import sys
    print(f"[TQ-TRITON] turboquant_fused_decode_gqa import failed: "
          f"{type(_e).__name__}: {_e}", file=sys.stderr, flush=True)
try:
    from turboquant.triton_kernels import turboquant_scores_gqa
    _SCORE_AVAILABLE = True
except (ImportError, Exception) as _e:
    import sys
    print(f"[TQ-TRITON] turboquant_scores_gqa import failed: "
          f"{type(_e).__name__}: {_e}", file=sys.stderr, flush=True)
try:
    from turboquant.triton_kernels import turboquant_fused_decode_graph
    _GRAPH_AVAILABLE = True
except (ImportError, Exception) as _e:
    _GRAPH_AVAILABLE = False
    import sys
    print(f"[TQ-TRITON] turboquant_fused_decode_graph import failed: "
          f"{type(_e).__name__}: {_e}", file=sys.stderr, flush=True)

if _GRAPH_AVAILABLE:
    _triton_path = "graph"
elif _FUSED_AVAILABLE:
    _triton_path = "fused"
elif _SCORE_AVAILABLE:
    _triton_path = "score"
else:
    _triton_path = "pytorch"

logger.info("[TurboQuant] Triton path: %s  graph=%s", _triton_path, _GRAPH_AVAILABLE)


# ===================================================================
# Pre-allocation setup
# ===================================================================

def preallocate_layer(state, max_tokens: int):
    """Pre-allocate all buffers for CUDA-Graph-compatible decode.

    Call once during model initialization, before any CUDA Graph capture.
    """
    store = state.store
    Q = state.config.num_query_heads
    D = state.config.head_dim

    # Pre-allocate compressed KV flat buffer
    store.preallocate(max_tokens)

    # Pre-allocate output buffers for fused kernel
    dev = state.config.device
    state._acc_buf = torch.zeros(Q, D, device=dev, dtype=torch.float32)
    state._m_buf = torch.zeros(Q, device=dev, dtype=torch.float32)
    state._l_buf = torch.zeros(Q, device=dev, dtype=torch.float32)

    # Pre-allocated query rotation buffers
    state._q_rot_buf = torch.zeros(Q, D, device=dev, dtype=torch.float32)
    state._q_sketch_buf = torch.zeros(Q, D, device=dev, dtype=torch.float32)

    # Pre-allocate ring buffer device tensors for CUDA-Graph-compatible writes
    state.engine.ring.preallocate_graph_buffers()

    logger.info(
        "[TurboQuant] Pre-allocated layer %d: max_tokens=%d Q=%d D=%d ring=%d (%.1f MB)",
        state.config.layer_idx, max_tokens, Q, D, state.engine.ring.capacity,
        (store.memory_bytes() + state._acc_buf.nelement() * 4 * 3) / 1e6,
    )


# ===================================================================
# Public entry point
# ===================================================================

def compute_hybrid_attention(
    query: torch.Tensor,
    store: CompressedKVStore,
    recent_k: Optional[torch.Tensor],
    recent_v: Optional[torch.Tensor],
    num_query_heads: int,
    scale: Optional[float] = None,
    layer_state=None,
) -> torch.Tensor:
    """Compute attention output combining compressed history and exact recent buffer.

    Args:
        query: (T, Q, D) — typically T=1 for decode
        store: compressed KV store
        recent_k: (recent_len, H_kv, D) or None
        recent_v: (recent_len, H_kv, D) or None
        num_query_heads: total query heads
        scale: 1/sqrt(head_dim)
        layer_state: LayerState (for pre-allocated buffers)

    Returns:
        output: (T, Q, D)
    """
    head_dim = store.head_dim
    num_kv_heads = store.num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    flat = store.get_flat_cache()
    has_history = flat is not None and (
        store.is_preallocated or flat.num_tokens >= MIN_HISTORY_FOR_TQ
    )
    has_recent = recent_k is not None and recent_k.shape[0] > 0

    if not has_history and not has_recent:
        return torch.zeros(
            query.shape[0], num_query_heads, head_dim,
            device=query.device, dtype=query.dtype,
        )

    gqa_ratio = num_query_heads // num_kv_heads

    if has_history and not has_recent:
        return _attend_compressed_only(
            query, flat, store, gqa_ratio, num_kv_heads, scale, layer_state,
        )
    if not has_history and has_recent:
        return _attend_exact_only(query, recent_k, recent_v, gqa_ratio, num_kv_heads, scale)

    return _attend_hybrid(
        query, flat, store, recent_k, recent_v,
        gqa_ratio, num_kv_heads, head_dim, scale, layer_state,
    )


# ===================================================================
# Compressed-only attention
# ===================================================================

def _attend_compressed_only(query, flat, store, gqa_ratio, num_kv_heads, scale, layer_state):
    global _diag_log_count
    _graph_intended = getattr(layer_state, 'graph_intended', store.is_preallocated)
    should_log = _diag_log_count < _DIAG_MAX_LOG and not _graph_intended
    if should_log:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    T, Q, D = query.shape

    # Prefer graph-compatible path when store is pre-allocated
    if store.is_preallocated and _GRAPH_AVAILABLE and T == 1:
        result = _compressed_graph(query, flat, store, gqa_ratio, scale, layer_state)
    elif _FUSED_AVAILABLE and T == 1:
        result = _compressed_fused(query, flat, store, gqa_ratio, scale)
    elif _SCORE_AVAILABLE and T == 1:
        result = _compressed_score_triton(query, flat, store, gqa_ratio, num_kv_heads, scale)
    else:
        result = _compressed_pytorch(query, flat, store, gqa_ratio, num_kv_heads, scale)

    if should_log:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        _diag_log_count += 1
        n = store._write_pos if store.is_preallocated else flat.num_tokens
        print(
            f"[TQ-SCORE] compressed_only path={_triton_path} "
            f"total={(t1-t0)*1000:.2f}ms N={n}",
            flush=True,
        )
    return result


# ===================================================================
# Hybrid attention (compressed + recent buffer)
# ===================================================================

def _attend_hybrid(query, flat, store, recent_k, recent_v,
                   gqa_ratio, num_kv_heads, head_dim, scale, layer_state):
    global _diag_log_count
    _graph_intended = getattr(layer_state, 'graph_intended', store.is_preallocated)
    should_log = _diag_log_count < _DIAG_MAX_LOG and not _graph_intended
    if should_log:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    T, Q, D = query.shape
    N_recent = recent_k.shape[0]

    if store.is_preallocated and _GRAPH_AVAILABLE and T == 1:
        result = _hybrid_graph(
            query, flat, store, recent_k, recent_v,
            gqa_ratio, num_kv_heads, scale, layer_state,
        )
    elif _FUSED_AVAILABLE and T == 1:
        result = _hybrid_fused(
            query, flat, store, recent_k, recent_v,
            gqa_ratio, num_kv_heads, scale,
        )
    elif _SCORE_AVAILABLE and T == 1:
        result = _hybrid_score_triton(
            query, flat, store, recent_k, recent_v,
            gqa_ratio, num_kv_heads, scale,
        )
    else:
        result = _hybrid_pytorch(
            query, flat, store, recent_k, recent_v,
            gqa_ratio, num_kv_heads, head_dim, scale,
        )

    if should_log:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        _diag_log_count += 1
        n = store._write_pos if store.is_preallocated else flat.num_tokens
        print(
            f"[TQ-SCORE] hybrid path={_triton_path} "
            f"total={(t1-t0)*1000:.2f}ms N_hist={n} N_recent={N_recent}",
            flush=True,
        )
    return result


def _attend_exact_only(query, recent_k, recent_v, gqa_ratio, num_kv_heads, scale):
    return _matmul_attend(
        query, recent_k.transpose(0, 1), recent_v.transpose(0, 1),
        gqa_ratio, num_kv_heads, scale,
    )


# ===================================================================
# Path 0: Graph-compatible fused kernel (CUDA Graph safe)
# ===================================================================

def _compressed_graph(query, flat, store, gqa_ratio, scale, layer_state):
    from turboquant.triton_kernels import turboquant_fused_decode_graph

    T, Q, D = query.shape
    q = query.squeeze(0).float()
    quantizer = store.quantizer

    # Pre-allocated query rotation: write into fixed-address buffer
    torch.matmul(q, quantizer.mse_quantizer.Pi.T, out=layer_state._q_rot_buf)
    torch.matmul(q, quantizer.S.T, out=layer_state._q_sketch_buf)

    acc, m, l = turboquant_fused_decode_graph(
        query=q,
        quantized_key=flat.prod_q,
        value_quantized=flat.value_q,
        Pi=quantizer.mse_quantizer.Pi,
        S=quantizer.S,
        centroids=quantizer.mse_quantizer.centroids,
        mse_bits=quantizer.mse_bits,
        qjl_scale=quantizer.qjl_scale,
        sm_scale=scale,
        gqa_ratio=gqa_ratio,
        acc_buf=layer_state._acc_buf,
        m_buf=layer_state._m_buf,
        l_buf=layer_state._l_buf,
        n_tensor=store.n_tensor,
    )
    out = acc / l.unsqueeze(-1)
    return out.unsqueeze(0).to(query.dtype)


def _hybrid_graph(query, flat, store, recent_k, recent_v,
                  gqa_ratio, num_kv_heads, scale, layer_state):
    from turboquant.triton_kernels import turboquant_fused_decode_graph

    T, Q, D = query.shape
    H_kv = num_kv_heads
    G = gqa_ratio
    quantizer = store.quantizer
    ring = layer_state.engine.ring

    q = query.squeeze(0).float()

    # Fused kernel on compressed KV
    torch.matmul(q, quantizer.mse_quantizer.Pi.T, out=layer_state._q_rot_buf)
    torch.matmul(q, quantizer.S.T, out=layer_state._q_sketch_buf)

    acc_c, m_c, l_c = turboquant_fused_decode_graph(
        query=q,
        quantized_key=flat.prod_q,
        value_quantized=flat.value_q,
        Pi=quantizer.mse_quantizer.Pi,
        S=quantizer.S,
        centroids=quantizer.mse_quantizer.centroids,
        mse_bits=quantizer.mse_bits,
        qjl_scale=quantizer.qjl_scale,
        sm_scale=scale,
        gqa_ratio=gqa_ratio,
        acc_buf=layer_state._acc_buf,
        m_buf=layer_state._m_buf,
        l_buf=layer_state._l_buf,
        n_tensor=store.n_tensor,
    )

    # Recent buffer — read full ring buffer + device count for CUDA-Graph compat
    if ring.graph_ready:
        ring_k, ring_v, count_tensor, arange_buf, cap_tensor = ring.peek_full()
        cap = ring.capacity
        k_r = ring_k.transpose(0, 1).float()  # (H_kv, cap, D)
        v_r = ring_v.transpose(0, 1).float()

        q_g = q.view(H_kv, G, D)
        scores_r = torch.bmm(q_g, k_r.transpose(1, 2)).reshape(Q, cap)

        # Mask invalid entries (entries beyond count are stale)
        valid_count = torch.minimum(count_tensor, cap_tensor)
        mask = arange_buf < valid_count  # (cap,) bool
        scores_r = scores_r.masked_fill(~mask.unsqueeze(0), float('-inf'))

        m_r = scores_r.max(dim=-1).values
        p_r = torch.exp(scores_r - m_r.unsqueeze(-1))
        l_r = p_r.sum(dim=-1)
        acc_r = torch.bmm(p_r.view(H_kv, G, cap), v_r).reshape(Q, D)
    else:
        # Non-graph fallback (shouldn't normally be called, but safe)
        N_recent = recent_k.shape[0]
        q_g = q.view(H_kv, G, D)
        k_r = recent_k.transpose(0, 1).float()
        v_r = recent_v.transpose(0, 1).float()
        scores_r = torch.bmm(q_g, k_r.transpose(1, 2)).reshape(Q, N_recent)
        m_r = scores_r.max(dim=-1).values
        p_r = torch.exp(scores_r - m_r.unsqueeze(-1))
        l_r = p_r.sum(dim=-1)
        acc_r = torch.bmm(p_r.view(H_kv, G, N_recent), v_r).reshape(Q, D)

    # Online softmax merge
    m = torch.maximum(m_c, m_r)
    alpha_c = torch.exp(m_c - m)
    alpha_r = torch.exp(m_r - m)
    l_merged = l_c * alpha_c + l_r * alpha_r
    acc_merged = acc_c * alpha_c.unsqueeze(-1) + acc_r * alpha_r.unsqueeze(-1)

    out = acc_merged / l_merged.unsqueeze(-1)
    return out.unsqueeze(0).to(query.dtype)


# ===================================================================
# Path 1: Fused Triton kernel (non-graph, creates tensors)
# ===================================================================

def _compressed_fused(query, flat, store, gqa_ratio, scale):
    from turboquant.triton_kernels import turboquant_fused_decode_gqa
    T, Q, D = query.shape
    q = query.squeeze(0).float()
    acc, m, l = turboquant_fused_decode_gqa(
        query=q, quantized_key=flat.prod_q, value_quantized=flat.value_q,
        Pi=store.quantizer.mse_quantizer.Pi, S=store.quantizer.S,
        centroids=store.quantizer.mse_quantizer.centroids,
        mse_bits=store.quantizer.mse_bits, qjl_scale=store.quantizer.qjl_scale,
        sm_scale=scale, gqa_ratio=gqa_ratio,
    )
    return (acc / l.unsqueeze(-1)).unsqueeze(0).to(query.dtype)


def _hybrid_fused(query, flat, store, recent_k, recent_v, gqa_ratio, num_kv_heads, scale):
    from turboquant.triton_kernels import turboquant_fused_decode_gqa
    T, Q, D = query.shape
    H_kv, G, N_recent = num_kv_heads, gqa_ratio, recent_k.shape[0]
    q = query.squeeze(0).float()

    acc_c, m_c, l_c = turboquant_fused_decode_gqa(
        query=q, quantized_key=flat.prod_q, value_quantized=flat.value_q,
        Pi=store.quantizer.mse_quantizer.Pi, S=store.quantizer.S,
        centroids=store.quantizer.mse_quantizer.centroids,
        mse_bits=store.quantizer.mse_bits, qjl_scale=store.quantizer.qjl_scale,
        sm_scale=scale, gqa_ratio=gqa_ratio,
    )

    q_g = q.view(H_kv, G, D)
    k_r = recent_k.transpose(0, 1).float()
    v_r = recent_v.transpose(0, 1).float()
    scores_r = torch.bmm(q_g, k_r.transpose(1, 2)).reshape(Q, N_recent)
    m_r = scores_r.max(dim=-1).values
    p_r = torch.exp(scores_r - m_r.unsqueeze(-1))
    l_r = p_r.sum(dim=-1)
    acc_r = torch.bmm(p_r.view(H_kv, G, N_recent), v_r).reshape(Q, D)

    m = torch.maximum(m_c, m_r)
    alpha_c, alpha_r = torch.exp(m_c - m), torch.exp(m_r - m)
    l_merged = l_c * alpha_c + l_r * alpha_r
    acc_merged = acc_c * alpha_c.unsqueeze(-1) + acc_r * alpha_r.unsqueeze(-1)
    return (acc_merged / l_merged.unsqueeze(-1)).unsqueeze(0).to(query.dtype)


# ===================================================================
# Path 2: Triton score kernels (K without dequant, V in PyTorch)
# ===================================================================

def _compressed_score_triton(query, flat, store, gqa_ratio, num_kv_heads, scale):
    from turboquant.triton_kernels import turboquant_scores_gqa
    T, Q, D = query.shape
    q = query.squeeze(0).float()
    scores = turboquant_scores_gqa(
        query=q, mse_packed=flat.prod_q.mse_indices,
        qjl_signs=flat.prod_q.qjl_signs, norms=flat.prod_q.norms,
        res_norms=flat.prod_q.residual_norms,
        centroids=store.quantizer.mse_quantizer.centroids,
        Pi=store.quantizer.mse_quantizer.Pi, S=store.quantizer.S,
        mse_bits=store.quantizer.mse_bits, qjl_scale=store.quantizer.qjl_scale,
        gqa_ratio=gqa_ratio,
    ).unsqueeze(0) * scale
    weights = F.softmax(scores.float(), dim=-1)
    v_dequant = dequantize_values(flat.value_q, 32)
    return _weighted_sum_gqa(weights, v_dequant.float(), num_kv_heads, gqa_ratio, query.dtype)


def _hybrid_score_triton(query, flat, store, recent_k, recent_v, gqa_ratio, num_kv_heads, scale):
    from turboquant.triton_kernels import turboquant_scores_gqa
    T, Q, D = query.shape
    N_hist, N_recent = flat.num_tokens, recent_k.shape[0]
    q = query.squeeze(0).float()

    scores_hist = turboquant_scores_gqa(
        query=q, mse_packed=flat.prod_q.mse_indices,
        qjl_signs=flat.prod_q.qjl_signs, norms=flat.prod_q.norms,
        res_norms=flat.prod_q.residual_norms,
        centroids=store.quantizer.mse_quantizer.centroids,
        Pi=store.quantizer.mse_quantizer.Pi, S=store.quantizer.S,
        mse_bits=store.quantizer.mse_bits, qjl_scale=store.quantizer.qjl_scale,
        gqa_ratio=gqa_ratio,
    ).unsqueeze(0)

    k_recent = recent_k.transpose(0, 1).float()
    q_g = query.float().view(T, num_kv_heads, gqa_ratio, D).permute(1, 2, 0, 3)
    scores_recent = torch.einsum("hgtd,hgnd->hgtn", q_g, k_recent.unsqueeze(1))
    scores_recent = scores_recent.permute(2, 0, 1, 3).reshape(T, Q, N_recent)

    scores_all = torch.cat([scores_hist, scores_recent], dim=-1) * scale
    weights = F.softmax(scores_all.float(), dim=-1)

    v_hist = dequantize_values(flat.value_q, 32)
    v_recent = recent_v.transpose(0, 1).float()
    v_all = torch.cat([v_hist, v_recent], dim=1)
    return _weighted_sum_gqa(weights, v_all, num_kv_heads, gqa_ratio, query.dtype)


# ===================================================================
# Path 3: PyTorch fallback
# ===================================================================

def _compressed_pytorch(query, flat, store, gqa_ratio, num_kv_heads, scale):
    k_dequant = store.quantizer.dequantize(flat.prod_q)
    v_dequant = dequantize_values(flat.value_q, 32)
    return _matmul_attend(query, k_dequant, v_dequant, gqa_ratio, num_kv_heads, scale)


def _hybrid_pytorch(query, flat, store, recent_k, recent_v, gqa_ratio, num_kv_heads, head_dim, scale):
    k_hist = store.quantizer.dequantize(flat.prod_q)
    v_hist = dequantize_values(flat.value_q, 32)
    k_recent, v_recent = recent_k.transpose(0, 1), recent_v.transpose(0, 1)
    k_all = torch.cat([k_hist.float(), k_recent.float()], dim=1)
    v_all = torch.cat([v_hist.float(), v_recent.float()], dim=1)
    return _matmul_attend(query, k_all, v_all, gqa_ratio, num_kv_heads, scale)


# ===================================================================
# Shared helpers
# ===================================================================

def _weighted_sum_gqa(weights, v_all, num_kv_heads, gqa_ratio, query_dtype):
    T, Q, N_total = weights.shape
    D = v_all.shape[-1]
    w = weights.view(T, num_kv_heads, gqa_ratio, N_total).permute(1, 2, 0, 3)
    v = v_all.unsqueeze(1)
    out = torch.einsum("hgtn,hgnd->hgtd", w, v)
    return out.permute(2, 0, 1, 3).reshape(T, Q, D).to(query_dtype)


def _matmul_attend(query, kv_keys, kv_values, gqa_ratio, num_kv_heads, scale):
    T, Q, D = query.shape
    q = query.float().view(T, num_kv_heads, gqa_ratio, D).permute(1, 2, 0, 3)
    k = kv_keys.float().unsqueeze(1)
    v = kv_values.float().unsqueeze(1)
    scores = torch.einsum("hgtd,hgnd->hgtn", q, k) * scale
    weights = F.softmax(scores, dim=-1)
    out = torch.einsum("hgtn,hgnd->hgtd", weights, v)
    return out.permute(2, 0, 1, 3).reshape(T, Q, D).to(query.dtype)
