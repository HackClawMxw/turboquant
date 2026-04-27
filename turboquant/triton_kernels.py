"""
TurboQuant fused Triton kernels for decode attention.

The main bottleneck during decode is computing attention scores from the
packed TurboQuant representation. Without fusion, the PyTorch path is:

  1. Unpack MSE indices (bit-shift)
  2. Lookup centroids (gather)
  3. Rotate back (d×d matmul)
  4. Scale by norms
  5. Dot with query (another matmul)
  ──
  6. Sketch query through S (d×d matmul)
  7. Unpack QJL signs (bit-shift)
  8. Dot sketched query with signs
  9. Scale by residual norms

With fusion, we avoid materializing the full d-dim dequantized vectors.
Instead we compute the score directly from packed data.

Kernel 1: turboquant_mse_score
  For each (query, quantized_key) pair, compute <q, dequant(key)>
  by fusing steps 1-5 into a single kernel.

Kernel 2: turboquant_qjl_score
  For each (query, qjl_signs) pair, compute <S^T q, signs> * scale
  by fusing steps 6-9 (query sketch is precomputed once per query).

Kernel 3: turboquant_fused_decode_attention
  Full fused kernel: computes softmax(scores/sqrt(d)) @ V in one pass
  using online softmax (flash-attention style) over TQ-compressed KV.
"""

import math
import torch
import triton
import triton.language as tl


# ─── Kernel 1: MSE score computation ──────────────────────────────────
#
# Given:
#   query:       (B*H, 1, D)           float16/float32
#   mse_packed:  (B*H, N, packed_d)    uint8 (bit-packed MSE indices)
#   norms:       (B*H, N)              float16/float32 (original vector norms)
#   centroids:   (2^mse_bits,)         float32 (codebook centroids)
#   Pi:          (D, D)                float32 (rotation matrix)
#
# Computes: scores[b,n] = sum_j query_rot[j] * centroid[idx[n,j]] * norms[n]
#
# Key insight: instead of rotating key back (y@Pi), we rotate query forward (q@Pi^T)
# Then score = norms * sum_j q_rot[j] * centroid[idx[j]]
# This avoids materializing the D-dim dequantized key vectors entirely.

@triton.jit
def _turboquant_mse_score_kernel(
    # Pointers
    Q_ptr,          # (BH, D) query vectors (already rotated: q @ Pi^T)
    MSE_ptr,        # (BH, N, packed_d) bit-packed indices
    NORMS_ptr,      # (BH, N) original norms
    CENTROIDS_ptr,  # (n_clusters,) centroid values
    OUT_ptr,        # (BH, N) output scores
    # Strides
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_n_bh, stride_n_n,
    stride_o_bh, stride_o_n,
    # Dimensions
    BH: tl.constexpr,
    N,   # number of KV tokens (variable)
    D: tl.constexpr,
    PACKED_D: tl.constexpr,
    # Quantization params
    BITS: tl.constexpr,        # MSE bits (1, 2, or 4 after rounding)
    VALS_PER_BYTE: tl.constexpr,  # how many indices per packed byte
    # Block sizes
    BLOCK_N: tl.constexpr,
):
    """Compute MSE attention scores for a block of KV tokens."""
    pid_bh = tl.program_id(0)   # batch*head index
    pid_n = tl.program_id(1)    # KV token block index

    # Bounds
    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Load the rotated query for this head: (D,)
    q_offs = tl.arange(0, D)
    q = tl.load(Q_ptr + pid_bh * stride_q_bh + q_offs * stride_q_d).to(tl.float32)

    # Accumulate score for each token in the block
    scores = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Bit mask for extracting indices
    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    # Process packed bytes — each byte contains VALS_PER_BYTE indices
    for byte_idx in range(PACKED_D):
        # Load packed bytes for this block of tokens: (BLOCK_N,)
        packed = tl.load(
            MSE_ptr + pid_bh * stride_m_bh + n_offs * stride_m_n + byte_idx * stride_m_d,
            mask=n_mask, other=0
        ).to(tl.int32)

        # Extract each index from the packed byte
        for sub in range(VALS_PER_BYTE):
            coord_idx = byte_idx * VALS_PER_BYTE + sub
            if coord_idx < D:
                # Extract index for this coordinate
                idx = (packed >> (sub * BITS)) & BIT_MASK

                # Lookup centroid value
                centroid_val = tl.load(CENTROIDS_ptr + idx)

                # Accumulate: q[coord_idx] * centroid[idx]
                q_val = tl.load(Q_ptr + pid_bh * stride_q_bh + coord_idx * stride_q_d).to(tl.float32)
                scores += q_val * centroid_val

    # Multiply by norms
    norms = tl.load(NORMS_ptr + pid_bh * stride_n_bh + n_offs * stride_n_n,
                     mask=n_mask, other=0.0).to(tl.float32)
    scores = scores * norms

    # Store
    tl.store(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n,
             scores, mask=n_mask)


# ─── Kernel 2: QJL score computation ──────────────────────────────────
#
# Given:
#   q_sketched:     (BH, D)         float32 — precomputed q @ S^T
#   qjl_signs:      (BH, N, D//8)   uint8 — packed sign bits
#   residual_norms: (BH, N)         float32
#   qjl_scale:      scalar          float32 — sqrt(pi/2) / D
#
# Computes: scores[b,n] = qjl_scale * res_norms[n] * sum_j q_sketched[j] * sign[n,j]

@triton.jit
def _turboquant_qjl_score_kernel(
    Q_SKETCH_ptr,    # (BH, D) pre-sketched query
    SIGNS_ptr,       # (BH, N, packed_d) packed sign bits
    RES_NORMS_ptr,   # (BH, N) residual norms
    OUT_ptr,         # (BH, N) output QJL scores (added to existing)
    # Strides
    stride_qs_bh, stride_qs_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_rn_bh, stride_rn_n,
    stride_o_bh, stride_o_n,
    # Dims
    N,
    D: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,  # D // 8
    QJL_SCALE,  # sqrt(pi/2) / D
    # Block sizes
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Accumulate dot product of q_sketched with sign vectors
    dot = tl.zeros([BLOCK_N], dtype=tl.float32)

    for byte_idx in range(PACKED_D_SIGNS):
        # Load packed sign byte for this block: (BLOCK_N,)
        packed = tl.load(
            SIGNS_ptr + pid_bh * stride_s_bh + n_offs * stride_s_n + byte_idx * stride_s_d,
            mask=n_mask, other=0
        ).to(tl.int32)

        # Extract 8 sign bits per byte
        for bit in range(8):
            coord_idx = byte_idx * 8 + bit
            if coord_idx < D:
                sign_bit = (packed >> bit) & 1
                # Convert {0,1} -> {-1, +1}
                sign_val = tl.where(sign_bit == 1, 1.0, -1.0)

                q_val = tl.load(Q_SKETCH_ptr + pid_bh * stride_qs_bh + coord_idx * stride_qs_d).to(tl.float32)
                dot += q_val * sign_val

    # Scale by residual norms and QJL constant
    res_norms = tl.load(RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n,
                         mask=n_mask, other=0.0).to(tl.float32)
    qjl_scores = dot * res_norms * QJL_SCALE

    # Add to existing MSE scores (or store fresh)
    existing = tl.load(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n,
                        mask=n_mask, other=0.0)
    tl.store(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n,
             existing + qjl_scores, mask=n_mask)


# ─── Kernel 3: Fused decode attention (online softmax over TQ keys + values) ──
#
# For decode, query has n_q=1. We iterate over KV tokens in blocks,
# computing scores from TQ-compressed keys and accumulating the
# weighted value sum using online softmax (flash-attention style).
#
# This is the big payoff: we read compressed KV (~3 bits/element),
# never materialize the full FP16 KV, and produce the final output
# in a single pass.

@triton.jit
def _turboquant_fused_decode_kernel(
    # Query (already rotated for MSE, and sketched for QJL)
    Q_ROT_ptr,       # (BH, D) q @ Pi^T
    Q_SKETCH_ptr,    # (BH, D) q @ S^T
    # Quantized keys
    MSE_ptr,         # (BH, N, packed_d_mse) packed MSE indices
    SIGNS_ptr,       # (BH, N, packed_d_signs) packed QJL signs
    NORMS_ptr,       # (BH, N) key norms
    RES_NORMS_ptr,   # (BH, N) residual norms
    CENTROIDS_ptr,   # (n_clusters,) codebook
    # Values (group-quantized)
    V_DATA_ptr,      # (BH, N, D) uint8 quantized values
    V_SCALES_ptr,    # (BH, N, N_GROUPS) value scales
    V_ZEROS_ptr,     # (BH, N, N_GROUPS) value zeros
    # Output
    OUT_ptr,         # (BH, D) output
    # Strides
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_n_bh, stride_n_n,
    stride_rn_bh, stride_rn_n,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_vs_bh, stride_vs_n, stride_vs_g,
    stride_vz_bh, stride_vz_n, stride_vz_g,
    stride_o_bh, stride_o_d,
    # Dims
    N,
    D: tl.constexpr,
    PACKED_D_MSE: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    N_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # Quant params
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    QJL_SCALE,
    SM_SCALE,  # 1/sqrt(d)
    # Block
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    # Online softmax state
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")  # running max
    l_i = tl.zeros([1], dtype=tl.float32)                   # running sum of exp
    acc = tl.zeros([D], dtype=tl.float32)                    # running weighted sum

    num_blocks = tl.cdiv(N, BLOCK_N)

    for block_idx in range(num_blocks):
        n_start = block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # ── Compute TQ attention score for this block ──

        # Part 1: MSE score
        mse_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_MSE):
            packed = tl.load(
                MSE_ptr + pid_bh * stride_m_bh + n_offs * stride_m_n + byte_idx * stride_m_d,
                mask=n_mask, other=0
            ).to(tl.int32)
            for sub in range(VALS_PER_BYTE):
                coord_idx = byte_idx * VALS_PER_BYTE + sub
                if coord_idx < D:
                    idx = (packed >> (sub * BITS)) & BIT_MASK
                    centroid_val = tl.load(CENTROIDS_ptr + idx)
                    q_val = tl.load(Q_ROT_ptr + pid_bh * stride_q_bh + coord_idx * stride_q_d).to(tl.float32)
                    mse_scores += q_val * centroid_val

        key_norms = tl.load(NORMS_ptr + pid_bh * stride_n_bh + n_offs * stride_n_n,
                            mask=n_mask, other=0.0).to(tl.float32)
        mse_scores = mse_scores * key_norms

        # Part 2: QJL score
        qjl_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_SIGNS):
            packed = tl.load(
                SIGNS_ptr + pid_bh * stride_s_bh + n_offs * stride_s_n + byte_idx * stride_s_d,
                mask=n_mask, other=0
            ).to(tl.int32)
            for bit in range(8):
                coord_idx = byte_idx * 8 + bit
                if coord_idx < D:
                    sign_bit = (packed >> bit) & 1
                    sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                    q_val = tl.load(Q_SKETCH_ptr + pid_bh * stride_q_bh + coord_idx * stride_q_d).to(tl.float32)
                    qjl_dot += q_val * sign_val

        res_norms = tl.load(RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n,
                            mask=n_mask, other=0.0).to(tl.float32)
        qjl_scores = qjl_dot * res_norms * QJL_SCALE

        # Combined score
        scores = (mse_scores + qjl_scores) * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        # ── Online softmax update ──
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        # Correction factor for previous accumulator
        alpha = tl.exp(m_i - m_new)
        # New exponentials
        p = tl.exp(scores - m_new)

        # Update running sum
        l_i = l_i * alpha + tl.sum(p, 0)
        # Update accumulator: rescale old, add new
        acc = acc * alpha

        # ── Dequantize values for this block and accumulate ──
        # Load full value tile: (BLOCK_N, D)
        d_offs = tl.arange(0, D)
        # Value data
        v_quant = tl.load(
            V_DATA_ptr + pid_bh * stride_v_bh
            + n_offs[:, None] * stride_v_n + d_offs[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0
        ).to(tl.float32)
        # Value scales: group index = d_offs // GROUP_SIZE
        g_offs = d_offs // GROUP_SIZE
        v_scale = tl.load(
            V_SCALES_ptr + pid_bh * stride_vs_bh
            + n_offs[:, None] * stride_vs_n + g_offs[None, :] * stride_vs_g,
            mask=n_mask[:, None], other=1.0
        ).to(tl.float32)
        v_zero = tl.load(
            V_ZEROS_ptr + pid_bh * stride_vz_bh
            + n_offs[:, None] * stride_vz_n + g_offs[None, :] * stride_vz_g,
            mask=n_mask[:, None], other=0.0
        ).to(tl.float32)
        # Dequantize: (BLOCK_N, D)
        v_dequant = v_quant * v_scale + v_zero
        # Weighted sum: p (BLOCK_N,) @ v_dequant (BLOCK_N, D) -> (D,)
        acc += tl.sum(p[:, None] * v_dequant, 0)

        m_i = m_new

    # Final normalization
    acc = acc / l_i

    # Store output
    d_offs = tl.arange(0, D)
    tl.store(OUT_ptr + pid_bh * stride_o_bh + d_offs * stride_o_d, acc)


# ─── Python wrappers ──────────────────────────────────────────────────

def _get_packing_params(bits: int):
    """Get packing parameters matching _pack_indices logic."""
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        return 4, 2  # 3-bit rounds up to 4-bit packing
    else:
        return 8, 1


def turboquant_mse_score(
    query_rot: torch.Tensor,     # (BH, D) or (BH, 1, D) — q @ Pi^T
    mse_packed: torch.Tensor,    # (BH, N, packed_d) uint8
    norms: torch.Tensor,         # (BH, N) float
    centroids: torch.Tensor,     # (n_clusters,) float32
    mse_bits: int,
) -> torch.Tensor:
    """
    Compute MSE attention scores using Triton kernel.

    Returns: (BH, N) attention logits (before scaling by 1/sqrt(d)).
    """
    if query_rot.dim() == 3:
        query_rot = query_rot.squeeze(1)  # (BH, D)

    BH, D = query_rot.shape
    N = mse_packed.shape[1]
    packed_d = mse_packed.shape[2]
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    out = torch.zeros(BH, N, device=query_rot.device, dtype=torch.float32)

    BLOCK_N = min(128, triton.next_power_of_2(N))

    grid = (BH, triton.cdiv(N, BLOCK_N))

    _turboquant_mse_score_kernel[grid](
        query_rot, mse_packed, norms, centroids, out,
        query_rot.stride(0), query_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        norms.stride(0), norms.stride(1),
        out.stride(0), out.stride(1),
        BH=BH, N=N, D=D, PACKED_D=packed_d,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        BLOCK_N=BLOCK_N,
    )

    return out


def turboquant_qjl_score(
    q_sketched: torch.Tensor,       # (BH, D) — q @ S^T
    qjl_signs: torch.Tensor,        # (BH, N, D//8) uint8 packed signs
    residual_norms: torch.Tensor,   # (BH, N)
    qjl_scale: float,               # sqrt(pi/2) / D
    out: torch.Tensor = None,       # (BH, N) — will be ADDED to if provided
) -> torch.Tensor:
    """
    Compute QJL attention score contribution.

    If `out` is provided, the QJL scores are added to it in-place.
    Returns: (BH, N) combined scores.
    """
    if q_sketched.dim() == 3:
        q_sketched = q_sketched.squeeze(1)

    BH, D = q_sketched.shape
    N = qjl_signs.shape[1]
    packed_d_signs = qjl_signs.shape[2]

    if out is None:
        out = torch.zeros(BH, N, device=q_sketched.device, dtype=torch.float32)

    BLOCK_N = min(128, triton.next_power_of_2(N))
    grid = (BH, triton.cdiv(N, BLOCK_N))

    _turboquant_qjl_score_kernel[grid](
        q_sketched, qjl_signs, residual_norms, out,
        q_sketched.stride(0), q_sketched.stride(1),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        residual_norms.stride(0), residual_norms.stride(1),
        out.stride(0), out.stride(1),
        N=N, D=D, PACKED_D_SIGNS=packed_d_signs,
        QJL_SCALE=qjl_scale,
        BLOCK_N=BLOCK_N,
    )

    return out


def turboquant_attention_score(
    query: torch.Tensor,               # (B, H, 1, D) or (BH, 1, D)
    quantized_key,                      # ProdQuantized namedtuple
    Pi: torch.Tensor,                   # (D, D) rotation matrix
    S: torch.Tensor,                    # (D, D) QJL matrix
    centroids: torch.Tensor,           # (n_clusters,) codebook
    mse_bits: int,
    qjl_scale: float,
) -> torch.Tensor:
    """
    High-level: compute TurboQuant attention scores using Triton kernels.

    Precomputes q_rot = q @ Pi^T and q_sketch = q @ S^T,
    then calls the two Triton kernels.

    Returns: (BH, N) raw logits (caller applies /sqrt(d) and softmax).
    """
    # Flatten batch/head dims
    if query.dim() == 4:
        B, H, Q, D = query.shape
        query_flat = query.reshape(B * H, Q, D)
    else:
        query_flat = query
        D = query.shape[-1]

    # Precompute rotated and sketched queries (one-time per decode step)
    q_rot = torch.matmul(query_flat.squeeze(1).float(), Pi.T)      # (BH, D)
    q_sketch = torch.matmul(query_flat.squeeze(1).float(), S.T)    # (BH, D)

    # Flatten quantized key batch dims
    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    if mse_packed.dim() == 4:
        BH_shape = mse_packed.shape[:2]
        BH = BH_shape[0] * BH_shape[1]
        mse_packed = mse_packed.reshape(BH, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH, *qjl_signs.shape[2:])
        norms = norms.reshape(BH, -1)
        res_norms = res_norms.reshape(BH, -1)

    # MSE scores
    scores = turboquant_mse_score(q_rot, mse_packed, norms, centroids, mse_bits)

    # Add QJL scores
    scores = turboquant_qjl_score(q_sketch, qjl_signs, res_norms, qjl_scale, out=scores)

    return scores


def turboquant_fused_decode(
    query: torch.Tensor,               # (BH, 1, D) or (BH, D)
    quantized_key,                      # ProdQuantized
    value_quantized,                    # ValueQuantized
    Pi: torch.Tensor,                   # (D, D)
    S: torch.Tensor,                    # (D, D)
    centroids: torch.Tensor,           # (n_clusters,)
    mse_bits: int,
    qjl_scale: float,
    sm_scale: float,
    group_size: int = 32,
) -> torch.Tensor:
    """
    Fully fused decode attention: scores + softmax + value aggregation.
    Single pass over compressed KV, flash-attention style online softmax.

    Returns: (BH, D) attention output.
    """
    if query.dim() == 3:
        query = query.squeeze(1)
    BH, D = query.shape

    q_rot = torch.matmul(query.float(), Pi.T)
    q_sketch = torch.matmul(query.float(), S.T)

    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    if mse_packed.dim() > 3:
        BH_shape = mse_packed.shape[:2]
        BH_actual = BH_shape[0] * BH_shape[1]
        mse_packed = mse_packed.reshape(BH_actual, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH_actual, *qjl_signs.shape[2:])
        norms = norms.reshape(BH_actual, -1)
        res_norms = res_norms.reshape(BH_actual, -1)

    N = mse_packed.shape[1]
    packed_d_mse = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]

    v_data = value_quantized.data
    v_scales = value_quantized.scales
    v_zeros = value_quantized.zeros

    # Unpack bit-packed values if needed (2-bit: 4 vals/byte, 4-bit: 2 vals/byte)
    v_bits = value_quantized.bits if len(value_quantized) > 3 else 2
    if v_bits == 2 and v_data.shape[-1] != D:
        from turboquant.kv_cache import unpack_values
        v_data = unpack_values(value_quantized)
        # v_data is now (..., N, D) uint8
    elif v_bits == 4 and v_data.shape[-1] != D:
        from turboquant.kv_cache import unpack_values
        v_data = unpack_values(value_quantized)

    if v_data.dim() > 3:
        v_data = v_data.reshape(BH, N, -1)
        v_scales = v_scales.reshape(BH, N, -1)
        v_zeros = v_zeros.reshape(BH, N, -1)

    N_GROUPS = D // group_size
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    out = torch.zeros(BH, D, device=query.device, dtype=torch.float32)

    BLOCK_N = min(64, triton.next_power_of_2(N))

    grid = (BH,)

    _turboquant_fused_decode_kernel[grid](
        q_rot, q_sketch,
        mse_packed, qjl_signs, norms, res_norms, centroids,
        v_data, v_scales, v_zeros,
        out,
        # Q strides
        q_rot.stride(0), q_rot.stride(1),
        # MSE strides
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        # Signs strides
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        # Norms strides
        norms.stride(0), norms.stride(1),
        # Res norms strides
        res_norms.stride(0), res_norms.stride(1),
        # Value strides
        v_data.stride(0), v_data.stride(1), v_data.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        # Out strides
        out.stride(0), out.stride(1),
        # Dims
        N=N, D=D, PACKED_D_MSE=packed_d_mse, PACKED_D_SIGNS=packed_d_signs,
        N_GROUPS=N_GROUPS, GROUP_SIZE=group_size,
        # Quant params
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        QJL_SCALE=qjl_scale, SM_SCALE=sm_scale,
        # Block
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    return out.to(query.dtype)


# ─── GQA-aware score kernels ──────────────────────────────────────────
#
# For GQA models, num_query_heads > num_kv_heads.  Each KV head serves
# G = num_query_heads / num_kv_heads query heads.  These kernels map each
# query head to its KV head via:  kv_head = query_head // GQA_RATIO
#
# This avoids expanding / duplicating KV data (which would waste memory)
# and lets each query head share the same compressed KV read.


@triton.jit
def _turboquant_mse_score_gqa_kernel(
    # Pointers
    Q_ptr,
    MSE_ptr,
    NORMS_ptr,
    CENTROIDS_ptr,
    OUT_ptr,
    # Strides
    stride_q_qh, stride_q_d,
    stride_m_kv, stride_m_n, stride_m_d,
    stride_n_kv, stride_n_n,
    stride_o_qh, stride_o_n,
    # Dimensions
    QH: tl.constexpr,
    N,
    D: tl.constexpr,
    PACKED_D: tl.constexpr,
    # Quantization params
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    # Block
    BLOCK_N: tl.constexpr,
):
    """GQA-aware MSE score: each query head maps to its KV head."""
    pid_q = tl.program_id(0)   # query head index
    pid_n = tl.program_id(1)   # KV token block index
    pid_kv = pid_q // GQA_RATIO  # KV head index

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Load the full rotated query vector once into registers
    d_offs = tl.arange(0, D)
    q = tl.load(Q_ptr + pid_q * stride_q_qh + d_offs * stride_q_d).to(tl.float32)

    scores = tl.zeros([BLOCK_N], dtype=tl.float32)
    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    for byte_idx in range(PACKED_D):
        packed = tl.load(
            MSE_ptr + pid_kv * stride_m_kv + n_offs * stride_m_n + byte_idx * stride_m_d,
            mask=n_mask, other=0,
        ).to(tl.int32)
        for sub in range(VALS_PER_BYTE):
            coord_idx = byte_idx * VALS_PER_BYTE + sub
            if coord_idx < D:
                idx = (packed >> (sub * BITS)) & BIT_MASK
                centroid_val = tl.load(CENTROIDS_ptr + idx)
                scores += q[coord_idx] * centroid_val

    norms = tl.load(
        NORMS_ptr + pid_kv * stride_n_kv + n_offs * stride_n_n,
        mask=n_mask, other=0.0,
    ).to(tl.float32)
    scores = scores * norms

    tl.store(
        OUT_ptr + pid_q * stride_o_qh + n_offs * stride_o_n,
        scores, mask=n_mask,
    )


@triton.jit
def _turboquant_qjl_score_gqa_kernel(
    Q_SKETCH_ptr,
    SIGNS_ptr,
    RES_NORMS_ptr,
    OUT_ptr,
    # Strides
    stride_qs_qh, stride_qs_d,
    stride_s_kv, stride_s_n, stride_s_d,
    stride_rn_kv, stride_rn_n,
    stride_o_qh, stride_o_n,
    # Dims
    N,
    D: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    QJL_SCALE,
    GQA_RATIO: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """GQA-aware QJL score: adds QJL contribution in-place."""
    pid_q = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_kv = pid_q // GQA_RATIO

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Load sketched query once into registers
    d_offs = tl.arange(0, D)
    q_s = tl.load(Q_SKETCH_ptr + pid_q * stride_qs_qh + d_offs * stride_qs_d).to(tl.float32)

    dot = tl.zeros([BLOCK_N], dtype=tl.float32)

    for byte_idx in range(PACKED_D_SIGNS):
        packed = tl.load(
            SIGNS_ptr + pid_kv * stride_s_kv + n_offs * stride_s_n + byte_idx * stride_s_d,
            mask=n_mask, other=0,
        ).to(tl.int32)
        for bit in range(8):
            coord_idx = byte_idx * 8 + bit
            if coord_idx < D:
                sign_bit = (packed >> bit) & 1
                sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                dot += q_s[coord_idx] * sign_val

    res_norms = tl.load(
        RES_NORMS_ptr + pid_kv * stride_rn_kv + n_offs * stride_rn_n,
        mask=n_mask, other=0.0,
    ).to(tl.float32)
    qjl_scores = dot * res_norms * QJL_SCALE

    existing = tl.load(
        OUT_ptr + pid_q * stride_o_qh + n_offs * stride_o_n,
        mask=n_mask, other=0.0,
    )
    tl.store(
        OUT_ptr + pid_q * stride_o_qh + n_offs * stride_o_n,
        existing + qjl_scores, mask=n_mask,
    )


def turboquant_scores_gqa(
    query: torch.Tensor,
    mse_packed: torch.Tensor,
    qjl_signs: torch.Tensor,
    norms: torch.Tensor,
    res_norms: torch.Tensor,
    centroids: torch.Tensor,
    Pi: torch.Tensor,
    S: torch.Tensor,
    mse_bits: int,
    qjl_scale: float,
    gqa_ratio: int,
) -> torch.Tensor:
    """Compute TQ attention scores with GQA support via Triton kernels.

    Avoids materializing dequantized K — scores are computed directly from
    packed / compressed data.  Uses kv_head indirection for GQA so that
    compressed KV data is read exactly once per KV head (not per query head).

    Args:
        query:       (num_query_heads, D) or (num_query_heads, 1, D) — raw query
        mse_packed:  (H_kv, N, packed_d) uint8  bit-packed MSE indices
        qjl_signs:   (H_kv, N, packed_d_signs) uint8  packed sign bits
        norms:       (H_kv, N) float  key L2 norms
        res_norms:   (H_kv, N) float  residual L2 norms
        centroids:   (n_clusters,) float32  codebook
        Pi:          (D, D) float32  rotation matrix
        S:           (D, D) float32  QJL projection matrix
        mse_bits:    bits per MSE index
        qjl_scale:   sqrt(pi/2) / D
        gqa_ratio:   num_query_heads / num_kv_heads

    Returns:
        scores: (num_query_heads, N) raw logits (before 1/sqrt(d) scaling).
    """
    if query.dim() == 3:
        query = query.squeeze(1)  # (QH, D)

    QH, D = query.shape
    N = mse_packed.shape[1]
    packed_d = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    # One-time matmuls: rotate & sketch the query (cheap, QH × D × D)
    q_rot = torch.matmul(query.float(), Pi.T)    # (QH, D)
    q_sketch = torch.matmul(query.float(), S.T)  # (QH, D)

    scores = torch.zeros(QH, N, device=query.device, dtype=torch.float32)

    BLOCK_N = min(128, triton.next_power_of_2(N))
    grid = (QH, triton.cdiv(N, BLOCK_N))

    _turboquant_mse_score_gqa_kernel[grid](
        q_rot, mse_packed, norms, centroids, scores,
        q_rot.stride(0), q_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        norms.stride(0), norms.stride(1),
        scores.stride(0), scores.stride(1),
        QH=QH, N=N, D=D, PACKED_D=packed_d,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        GQA_RATIO=gqa_ratio,
        BLOCK_N=BLOCK_N,
    )

    _turboquant_qjl_score_gqa_kernel[grid](
        q_sketch, qjl_signs, res_norms, scores,
        q_sketch.stride(0), q_sketch.stride(1),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        res_norms.stride(0), res_norms.stride(1),
        scores.stride(0), scores.stride(1),
        N=N, D=D, PACKED_D_SIGNS=packed_d_signs,
        QJL_SCALE=qjl_scale,
        GQA_RATIO=gqa_ratio,
        BLOCK_N=BLOCK_N,
    )

    return scores


# ─── GQA-aware Fused Decode Kernel ────────────────────────────────────
#
# Single-pass kernel: TQ scores + online softmax + value dequant +
# weighted-sum — one kernel per query head.  Never materialises the full
# (N, D) dequantized K/V tensors.
#
# Returns UNNORMALISED (acc, m, l) so the caller can merge with a recent
# buffer segment before normalising.
#
# GQA: pid_kv = pid_q // GQA_RATIO


@triton.jit
def _turboquant_fused_decode_gqa_kernel(
    # Query (pre-rotated / pre-sketched)
    Q_ROT_ptr,
    Q_SKETCH_ptr,
    # Quantized keys
    MSE_ptr,
    SIGNS_ptr,
    NORMS_ptr,
    RES_NORMS_ptr,
    CENTROIDS_ptr,
    # Values (group-quantized, unpacked to per-element uint8)
    V_DATA_ptr,
    V_SCALES_ptr,
    V_ZEROS_ptr,
    # Outputs: unnormalised accumulator, running max, running sum
    OUT_ptr,
    M_OUT_ptr,
    L_OUT_ptr,
    # --- strides ---
    stride_q_qh, stride_q_d,
    stride_m_kv, stride_m_n, stride_m_d,
    stride_s_kv, stride_s_n, stride_s_d,
    stride_n_kv, stride_n_n,
    stride_rn_kv, stride_rn_n,
    stride_v_kv, stride_v_n, stride_v_d,
    stride_vs_kv, stride_vs_n, stride_vs_g,
    stride_vz_kv, stride_vz_n, stride_vz_g,
    stride_o_qh, stride_o_d,
    stride_m_qh,
    stride_l_qh,
    # --- dims ---
    N,
    D: tl.constexpr,
    PACKED_D_MSE: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    N_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # --- quant params ---
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    QJL_SCALE,
    SM_SCALE,
    GQA_RATIO: tl.constexpr,
    # --- block ---
    BLOCK_N: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_kv = pid_q // GQA_RATIO

    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    # Load query vectors once into registers
    d_offs = tl.arange(0, D)
    q_rot = tl.load(Q_ROT_ptr + pid_q * stride_q_qh + d_offs * stride_q_d).to(tl.float32)
    q_sketch = tl.load(Q_SKETCH_ptr + pid_q * stride_q_qh + d_offs * stride_q_d).to(tl.float32)

    # Online softmax state
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    for block_idx in range(tl.cdiv(N, BLOCK_N)):
        n_start = block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # ── MSE score ──
        mse_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_MSE):
            packed = tl.load(
                MSE_ptr + pid_kv * stride_m_kv + n_offs * stride_m_n + byte_idx * stride_m_d,
                mask=n_mask, other=0,
            ).to(tl.int32)
            for sub in range(VALS_PER_BYTE):
                coord_idx = byte_idx * VALS_PER_BYTE + sub
                if coord_idx < D:
                    idx = (packed >> (sub * BITS)) & BIT_MASK
                    centroid_val = tl.load(CENTROIDS_ptr + idx)
                    mse_scores += q_rot[coord_idx] * centroid_val

        key_norms = tl.load(
            NORMS_ptr + pid_kv * stride_n_kv + n_offs * stride_n_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        mse_scores = mse_scores * key_norms

        # ── QJL score ──
        qjl_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_SIGNS):
            packed = tl.load(
                SIGNS_ptr + pid_kv * stride_s_kv + n_offs * stride_s_n + byte_idx * stride_s_d,
                mask=n_mask, other=0,
            ).to(tl.int32)
            for bit in range(8):
                coord_idx = byte_idx * 8 + bit
                if coord_idx < D:
                    sign_bit = (packed >> bit) & 1
                    sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                    qjl_dot += q_sketch[coord_idx] * sign_val

        res_norms = tl.load(
            RES_NORMS_ptr + pid_kv * stride_rn_kv + n_offs * stride_rn_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        qjl_scores = qjl_dot * res_norms * QJL_SCALE

        # Combined score
        scores = (mse_scores + qjl_scores) * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        # ── Online softmax update ──
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        # ── Value dequantize + accumulate ──
        v_quant = tl.load(
            V_DATA_ptr + pid_kv * stride_v_kv
            + n_offs[:, None] * stride_v_n + d_offs[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.float32)
        g_offs = d_offs // GROUP_SIZE
        v_scale = tl.load(
            V_SCALES_ptr + pid_kv * stride_vs_kv
            + n_offs[:, None] * stride_vs_n + g_offs[None, :] * stride_vs_g,
            mask=n_mask[:, None], other=1.0,
        ).to(tl.float32)
        v_zero = tl.load(
            V_ZEROS_ptr + pid_kv * stride_vz_kv
            + n_offs[:, None] * stride_vz_n + g_offs[None, :] * stride_vz_g,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)
        v_dequant = v_quant * v_scale + v_zero
        acc += tl.sum(p[:, None] * v_dequant, 0)

        m_i = m_new

    # Store unnormalised output + softmax state
    tl.store(OUT_ptr + pid_q * stride_o_qh + d_offs * stride_o_d, acc)
    tl.store(M_OUT_ptr + pid_q * stride_m_qh, tl.sum(m_i))
    tl.store(L_OUT_ptr + pid_q * stride_l_qh, tl.sum(l_i))


def turboquant_fused_decode_gqa(
    query: torch.Tensor,
    quantized_key,
    value_quantized,
    Pi: torch.Tensor,
    S: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    qjl_scale: float,
    sm_scale: float,
    gqa_ratio: int,
    group_size: int = 32,
):
    """Fully fused decode attention with GQA support.

    Single kernel per query head: scores + online softmax + value aggregation.
    Reads compressed KV directly — never materialises dequantized K/V tensors.

    Returns:
        acc: (QH, D) unnormalised weighted-sum
        m:   (QH,)   running max of scaled scores
        l:   (QH,)   running sum of exponentials
    """
    if query.dim() == 3:
        query = query.squeeze(1)
    QH, D = query.shape

    # Precompute rotated & sketched queries (two small matmuls)
    q_rot = torch.matmul(query.float(), Pi.T)    # (QH, D)
    q_sketch = torch.matmul(query.float(), S.T)  # (QH, D)

    # Flatten quantized key batch dims
    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    if mse_packed.dim() > 3:
        BH_actual = mse_packed.shape[0] * mse_packed.shape[1]
        mse_packed = mse_packed.reshape(BH_actual, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH_actual, *qjl_signs.shape[2:])
        norms = norms.reshape(BH_actual, -1)
        res_norms = res_norms.reshape(BH_actual, -1)

    N = mse_packed.shape[1]
    packed_d_mse = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]

    # Unpack bit-packed values to per-element uint8
    v_data = value_quantized.data
    v_scales = value_quantized.scales
    v_zeros = value_quantized.zeros
    v_bits = value_quantized.bits if len(value_quantized) > 3 else 2
    if v_bits == 2 and v_data.shape[-1] != D:
        from turboquant.kv_cache import unpack_values
        v_data = unpack_values(value_quantized)
    elif v_bits == 4 and v_data.shape[-1] != D:
        from turboquant.kv_cache import unpack_values
        v_data = unpack_values(value_quantized)

    if v_data.dim() > 3:
        H_kv = mse_packed.shape[0]
        v_data = v_data.reshape(H_kv, N, -1)
        v_scales = v_scales.reshape(H_kv, N, -1)
        v_zeros = v_zeros.reshape(H_kv, N, -1)

    N_GROUPS = D // group_size
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    # Output tensors
    acc = torch.zeros(QH, D, device=query.device, dtype=torch.float32)
    m_out = torch.zeros(QH, device=query.device, dtype=torch.float32)
    l_out = torch.zeros(QH, device=query.device, dtype=torch.float32)

    BLOCK_N = min(64, triton.next_power_of_2(N))
    grid = (QH,)

    _turboquant_fused_decode_gqa_kernel[grid](
        q_rot, q_sketch,
        mse_packed, qjl_signs, norms, res_norms, centroids,
        v_data, v_scales, v_zeros,
        acc, m_out, l_out,
        # Q strides (same for q_rot and q_sketch — both contiguous)
        q_rot.stride(0), q_rot.stride(1),
        # MSE strides
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        # Signs strides
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        # Norms strides
        norms.stride(0), norms.stride(1),
        # Res norms strides
        res_norms.stride(0), res_norms.stride(1),
        # Value strides
        v_data.stride(0), v_data.stride(1), v_data.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        # Output strides
        acc.stride(0), acc.stride(1),
        m_out.stride(0),
        l_out.stride(0),
        # Dims
        N=N, D=D, PACKED_D_MSE=packed_d_mse, PACKED_D_SIGNS=packed_d_signs,
        N_GROUPS=N_GROUPS, GROUP_SIZE=group_size,
        # Quant params
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        QJL_SCALE=qjl_scale, SM_SCALE=sm_scale,
        GQA_RATIO=gqa_ratio,
        # Block
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    return acc, m_out, l_out


# ─── CUDA-Graph-compatible Fused Kernel ─────────────────────────────
#
# Reads N from a device int32 pointer at runtime (while loop), so the
# graph replay uses the current N without recapture.  All input/output
# tensors must be pre-allocated at fixed addresses.


@triton.jit
def _turboquant_fused_decode_graph_kernel(
    Q_ROT_ptr,
    Q_SKETCH_ptr,
    MSE_ptr,
    SIGNS_ptr,
    NORMS_ptr,
    RES_NORMS_ptr,
    CENTROIDS_ptr,
    V_DATA_ptr,
    V_SCALES_ptr,
    V_ZEROS_ptr,
    OUT_ptr,
    M_OUT_ptr,
    L_OUT_ptr,
    N_PTR,
    # --- strides ---
    stride_q_qh, stride_q_d,
    stride_m_kv, stride_m_n, stride_m_d,
    stride_s_kv, stride_s_n, stride_s_d,
    stride_n_kv, stride_n_n,
    stride_rn_kv, stride_rn_n,
    stride_v_kv, stride_v_n, stride_v_d,
    stride_vs_kv, stride_vs_n, stride_vs_g,
    stride_vz_kv, stride_vz_n, stride_vz_g,
    stride_o_qh, stride_o_d,
    stride_m_qh,
    stride_l_qh,
    # --- constexpr dims ---
    D: tl.constexpr,
    PACKED_D_MSE: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    N_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    QJL_SCALE,
    SM_SCALE,
    GQA_RATIO: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_kv = pid_q // GQA_RATIO

    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    # Read N dynamically from device memory
    N = tl.load(N_PTR).to(tl.int32)

    # Load query vectors: store base pointers for element-wise access
    # (indexing a loaded tensor with a Python int inside nested loops
    #  triggers a Triton 3.x compilation bug, so we load per-element)
    q_rot_base = Q_ROT_ptr + pid_q * stride_q_qh
    q_sketch_base = Q_SKETCH_ptr + pid_q * stride_q_qh
    d_offs = tl.arange(0, D)  # needed for value dequant and output store

    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    # While loop: dynamic N read from device memory
    block_idx = 0
    while block_idx * BLOCK_N < N:
        n_start = block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # ── MSE score ──
        mse_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_MSE):
            packed = tl.load(
                MSE_ptr + pid_kv * stride_m_kv + n_offs * stride_m_n + byte_idx * stride_m_d,
                mask=n_mask, other=0,
            ).to(tl.int32)
            for sub in range(VALS_PER_BYTE):
                coord_idx = byte_idx * VALS_PER_BYTE + sub
                if coord_idx < D:
                    idx = (packed >> (sub * BITS)) & BIT_MASK
                    centroid_val = tl.load(CENTROIDS_ptr + idx)
                    q_val = tl.load(q_rot_base + coord_idx * stride_q_d).to(tl.float32)
                    mse_scores += q_val * centroid_val

        key_norms = tl.load(
            NORMS_ptr + pid_kv * stride_n_kv + n_offs * stride_n_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        mse_scores = mse_scores * key_norms

        # ── QJL score ──
        qjl_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_SIGNS):
            packed = tl.load(
                SIGNS_ptr + pid_kv * stride_s_kv + n_offs * stride_s_n + byte_idx * stride_s_d,
                mask=n_mask, other=0,
            ).to(tl.int32)
            for bit in range(8):
                coord_idx = byte_idx * 8 + bit
                if coord_idx < D:
                    sign_bit = (packed >> bit) & 1
                    sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                    s_val = tl.load(q_sketch_base + coord_idx * stride_q_d).to(tl.float32)
                    qjl_dot += s_val * sign_val

        res_norms = tl.load(
            RES_NORMS_ptr + pid_kv * stride_rn_kv + n_offs * stride_rn_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        qjl_scores = qjl_dot * res_norms * QJL_SCALE

        scores = (mse_scores + qjl_scores) * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        # ── Online softmax update ──
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        # ── Value dequantize + accumulate ──
        v_quant = tl.load(
            V_DATA_ptr + pid_kv * stride_v_kv
            + n_offs[:, None] * stride_v_n + d_offs[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.float32)
        g_offs = d_offs // GROUP_SIZE
        v_scale = tl.load(
            V_SCALES_ptr + pid_kv * stride_vs_kv
            + n_offs[:, None] * stride_vs_n + g_offs[None, :] * stride_vs_g,
            mask=n_mask[:, None], other=1.0,
        ).to(tl.float32)
        v_zero = tl.load(
            V_ZEROS_ptr + pid_kv * stride_vz_kv
            + n_offs[:, None] * stride_vz_n + g_offs[None, :] * stride_vz_g,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)
        v_dequant = v_quant * v_scale + v_zero
        acc += tl.sum(p[:, None] * v_dequant, 0)

        m_i = m_new
        block_idx += 1

    tl.store(OUT_ptr + pid_q * stride_o_qh + d_offs * stride_o_d, acc)
    tl.store(M_OUT_ptr + pid_q * stride_m_qh, tl.sum(m_i))
    tl.store(L_OUT_ptr + pid_q * stride_l_qh, tl.sum(l_i))


# ─── Fused recent-buffer decode kernel ────────────────────────────────
#
# Computes attention over a ring buffer of exact (non-quantized) KV pairs.
# Each program handles one query head, iterates over the ring buffer in
# blocks, applies count-based masking, online softmax, and value
# accumulation.  Reads bf16 ring buffer directly, casts to float32
# on-the-fly — no host-side tensor allocation.
#

@triton.jit
def _turboquant_recent_buffer_kernel(
    Q_ptr,            # (QH, D) float32 query
    RING_K_ptr,       # (cap, H_kv, D) bf16 ring buffer keys
    RING_V_ptr,       # (cap, H_kv, D) bf16 ring buffer values
    COUNT_ptr,        # (1,) int64 — valid entry count
    ARANGE_ptr,       # (cap,) int64 — [0, 1, ..., cap-1]
    CAP_ptr,          # (1,) int64 — ring buffer capacity
    OUT_ACC_ptr,      # (QH, D) float32 — output accumulator
    OUT_M_ptr,        # (QH,) float32 — output max
    OUT_L_ptr,        # (QH,) float32 — output sum-exp
    # --- strides ---
    stride_q_qh, stride_q_d,
    stride_k_cap, stride_k_h, stride_k_d,
    stride_v_cap, stride_v_h, stride_v_d,
    stride_o_qh, stride_o_d,
    stride_m_qh,
    stride_l_qh,
    # --- constexpr dims ---
    D: tl.constexpr,
    CAP: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SM_SCALE,
):
    pid_q = tl.program_id(0)
    pid_kv = pid_q // GQA_RATIO

    # Read valid count from device memory
    valid_count = tl.load(COUNT_ptr).to(tl.int32)
    cap_val = tl.load(CAP_ptr).to(tl.int32)
    actual_count = tl.minimum(valid_count, cap_val)

    # Query base pointer for per-element loads
    q_base = Q_ptr + pid_q * stride_q_qh
    d_offs = tl.arange(0, D)

    # Online softmax state
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    # Iterate over ring buffer in blocks
    for block_start in range(0, CAP, BLOCK_N):
        n_offs = block_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < actual_count

        # Load bf16 keys and cast to float32: (BLOCK_N, D)
        k_bf16 = tl.load(
            RING_K_ptr + n_offs[:, None] * stride_k_cap + pid_kv * stride_k_h + d_offs[None, :] * stride_k_d,
            mask=n_mask[:, None], other=0,
        )
        k_f32 = k_bf16.to(tl.float32)

        # Compute dot product with query: (BLOCK_N,)
        scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for d_idx in range(D):
            q_val = tl.load(q_base + d_idx * stride_q_d).to(tl.float32)
            scores += q_val * k_f32[:, d_idx]
        scores = scores * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        # Load bf16 values and cast to float32: (BLOCK_N, D)
        v_bf16 = tl.load(
            RING_V_ptr + n_offs[:, None] * stride_v_cap + pid_kv * stride_v_h + d_offs[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0,
        )
        v_f32 = v_bf16.to(tl.float32)

        # Weighted accumulation
        acc += tl.sum(p[:, None] * v_f32, 0)
        m_i = m_new

    tl.store(OUT_ACC_ptr + pid_q * stride_o_qh + d_offs * stride_o_d, acc)
    tl.store(OUT_M_ptr + pid_q * stride_m_qh, tl.sum(m_i))
    tl.store(OUT_L_ptr + pid_q * stride_l_qh, tl.sum(l_i))


def turboquant_recent_buffer_decode(
    query: torch.Tensor,       # (QH, D) float32
    ring_k: torch.Tensor,      # (cap, H_kv, D) bf16
    ring_v: torch.Tensor,      # (cap, H_kv, D) bf16
    count_tensor: torch.Tensor, # (1,) int64
    arange_buf: torch.Tensor,  # (cap,) int64
    cap_tensor: torch.Tensor,  # (1,) int64
    sm_scale: float,
    gqa_ratio: int,
    # Pre-allocated output buffers
    acc_buf: torch.Tensor = None,
    m_buf: torch.Tensor = None,
    l_buf: torch.Tensor = None,
):
    """Fused recent-buffer attention using Triton.

    Computes attention scores between query and ring buffer KV pairs,
    applies count-based masking, online softmax, and value accumulation
    in a single kernel.  No intermediate tensor allocations.

    Returns (acc, m, l) into pre-allocated buffers.
    """
    QH, D = query.shape
    cap = ring_k.shape[0]

    if acc_buf is None:
        acc_buf = torch.zeros(QH, D, device=query.device, dtype=torch.float32)
    if m_buf is None:
        m_buf = torch.zeros(QH, device=query.device, dtype=torch.float32)
    if l_buf is None:
        l_buf = torch.zeros(QH, device=query.device, dtype=torch.float32)

    acc_buf.zero_()
    m_buf.fill_(float("-inf"))
    l_buf.zero_()

    BLOCK_N = 32

    grid = (QH,)

    _turboquant_recent_buffer_kernel[grid](
        query, ring_k, ring_v,
        count_tensor, arange_buf, cap_tensor,
        acc_buf, m_buf, l_buf,
        query.stride(0), query.stride(1),
        ring_k.stride(0), ring_k.stride(1), ring_k.stride(2),
        ring_v.stride(0), ring_v.stride(1), ring_v.stride(2),
        acc_buf.stride(0), acc_buf.stride(1),
        m_buf.stride(0),
        l_buf.stride(0),
        D=D, CAP=cap, GQA_RATIO=gqa_ratio,
        BLOCK_N=BLOCK_N, SM_SCALE=sm_scale,
    )

    return acc_buf, m_buf, l_buf


# ─── Hybrid merge kernel ─────────────────────────────────────────────

@triton.jit
def _turboquant_hybrid_merge_kernel(
    ACC_C_ptr,    # (QH, D) float32
    M_C_ptr,      # (QH,) float32
    L_C_ptr,      # (QH,) float32
    ACC_R_ptr,    # (QH, D) float32
    M_R_ptr,      # (QH,) float32
    L_R_ptr,      # (QH,) float32
    OUT_ptr,      # (QH, D) float32 output
    stride_a_qh, stride_a_d,
    stride_m_qh,
    stride_l_qh,
    stride_o_qh, stride_o_d,
    D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    d_offs = tl.arange(0, D)

    m_c = tl.load(M_C_ptr + pid_q * stride_m_qh)
    l_c = tl.load(L_C_ptr + pid_q * stride_l_qh)
    m_r = tl.load(M_R_ptr + pid_q * stride_m_qh)
    l_r = tl.load(L_R_ptr + pid_q * stride_l_qh)

    m_merged = tl.maximum(m_c, m_r)
    alpha_c = tl.exp(m_c - m_merged)
    alpha_r = tl.exp(m_r - m_merged)
    l_merged = l_c * alpha_c + l_r * alpha_r

    acc_c = tl.load(ACC_C_ptr + pid_q * stride_a_qh + d_offs * stride_a_d)
    acc_r = tl.load(ACC_R_ptr + pid_q * stride_a_qh + d_offs * stride_a_d)

    acc_merged = acc_c * alpha_c + acc_r * alpha_r
    out = acc_merged / l_merged

    tl.store(OUT_ptr + pid_q * stride_o_qh + d_offs * stride_o_d, out)


def turboquant_hybrid_merge(
    acc_c: torch.Tensor,
    m_c: torch.Tensor,
    l_c: torch.Tensor,
    acc_r: torch.Tensor,
    m_r: torch.Tensor,
    l_r: torch.Tensor,
    out_buf: torch.Tensor = None,
) -> torch.Tensor:
    """Merge online softmax states from compressed and recent attention."""
    QH, D = acc_c.shape

    if out_buf is None:
        out_buf = torch.zeros(QH, D, device=acc_c.device, dtype=torch.float32)

    grid = (QH,)
    _turboquant_hybrid_merge_kernel[grid](
        acc_c, m_c, l_c,
        acc_r, m_r, l_r,
        out_buf,
        acc_c.stride(0), acc_c.stride(1),
        m_c.stride(0),
        l_c.stride(0),
        out_buf.stride(0), out_buf.stride(1),
        D=D,
    )

    return out_buf


def turboquant_fused_decode_graph(
    query: torch.Tensor,
    quantized_key,
    value_quantized,
    Pi: torch.Tensor,
    S: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    qjl_scale: float,
    sm_scale: float,
    gqa_ratio: int,
    group_size: int = 32,
    # Pre-allocated output buffers (CUDA Graph compatible)
    acc_buf: torch.Tensor = None,
    m_buf: torch.Tensor = None,
    l_buf: torch.Tensor = None,
    # Device-side N counter
    n_tensor: torch.Tensor = None,
    # Pre-computed query rotation (avoids redundant matmul)
    q_rot: torch.Tensor = None,
    q_sketch: torch.Tensor = None,
):
    """CUDA-Graph-compatible fused decode with GQA.

    Reads N from ``n_tensor`` device memory at runtime (while loop).
    Writes into pre-allocated ``acc_buf``, ``m_buf``, ``l_buf``.
    All tensor addresses remain stable across calls.

    Returns (acc, m, l) — the SAME tensors passed in as acc_buf/m_buf/l_buf.
    """
    if query.dim() == 3:
        query = query.squeeze(1)
    QH, D = query.shape

    # Use pre-computed rotations if provided (avoids redundant matmul)
    if q_rot is None:
        q_rot = torch.matmul(query.float(), Pi.T)
    if q_sketch is None:
        q_sketch = torch.matmul(query.float(), S.T)

    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    if mse_packed.dim() > 3:
        BH_actual = mse_packed.shape[0] * mse_packed.shape[1]
        mse_packed = mse_packed.reshape(BH_actual, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH_actual, *qjl_signs.shape[2:])
        norms = norms.reshape(BH_actual, -1)
        res_norms = res_norms.reshape(BH_actual, -1)

    packed_d_mse = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]

    v_data = value_quantized.data
    v_scales = value_quantized.scales
    v_zeros = value_quantized.zeros
    v_bits = value_quantized.bits if len(value_quantized) > 3 else 2
    if v_bits == 2 and v_data.shape[-1] != D:
        from turboquant.kv_cache import unpack_values
        v_data = unpack_values(value_quantized)
    elif v_bits == 4 and v_data.shape[-1] != D:
        from turboquant.kv_cache import unpack_values
        v_data = unpack_values(value_quantized)
    if v_data.dim() > 3:
        H_kv = mse_packed.shape[0]
        v_data = v_data.reshape(H_kv, -1, D)
        v_scales = v_scales.reshape(H_kv, -1, -1)
        v_zeros = v_zeros.reshape(H_kv, -1, -1)

    N_GROUPS = D // group_size
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    if acc_buf is None:
        acc_buf = torch.zeros(QH, D, device=query.device, dtype=torch.float32)
    if m_buf is None:
        m_buf = torch.zeros(QH, device=query.device, dtype=torch.float32)
    if l_buf is None:
        l_buf = torch.zeros(QH, device=query.device, dtype=torch.float32)
    # Zero out output buffers
    acc_buf.zero_()
    m_buf.zero_()
    l_buf.zero_()

    BLOCK_N = 64  # fixed for CUDA Graph
    grid = (QH,)

    _turboquant_fused_decode_graph_kernel[grid](
        q_rot, q_sketch,
        mse_packed, qjl_signs, norms, res_norms, centroids,
        v_data, v_scales, v_zeros,
        acc_buf, m_buf, l_buf,
        n_tensor,
        q_rot.stride(0), q_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        norms.stride(0), norms.stride(1),
        res_norms.stride(0), res_norms.stride(1),
        v_data.stride(0), v_data.stride(1), v_data.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        acc_buf.stride(0), acc_buf.stride(1),
        m_buf.stride(0),
        l_buf.stride(0),
        D=D, PACKED_D_MSE=packed_d_mse, PACKED_D_SIGNS=packed_d_signs,
        N_GROUPS=N_GROUPS, GROUP_SIZE=group_size,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        QJL_SCALE=qjl_scale, SM_SCALE=sm_scale,
        GQA_RATIO=gqa_ratio,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    return acc_buf, m_buf, l_buf
