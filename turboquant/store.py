"""
TurboQuant compressed KV store — CUDA-Graph-compatible design.

Design rules:
  - All compressed KV data lives in pre-allocated fixed-address buffers.
  - N (number of valid tokens) is stored in a device int32 tensor, updated
    each decode step.  The Triton kernel reads it at runtime via a while loop.
  - No dynamic tensor allocation on the hot path.
  - Chunk writes go directly into the pre-allocated flat buffer.
"""

from __future__ import annotations

import torch
from typing import Optional, NamedTuple

from turboquant.quantizer import TurboQuantProd, ProdQuantized
from turboquant.kv_cache import quantize_values, unpack_values, ValueQuantized


class FlatCache(NamedTuple):
    """Flattened view of compressed KV for fast read access."""
    prod_q: ProdQuantized       # (num_kv_heads, max_tokens, ...)
    value_q: ValueQuantized     # (num_kv_heads, max_tokens, ...)
    num_tokens: int             # logical count (actual valid tokens)


class CompressedKVStore:
    """CUDA-Graph-compatible compressed KV store.

    All tensors are pre-allocated at a fixed max capacity.
    ``n_tensor`` is a (1,) int32 device tensor holding the current valid count.
    Writers increment it; the Triton kernel reads it at runtime.
    """

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        device: torch.device = None,
        layer_idx: int = 0,
        max_tokens: int = 0,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = min(value_group_size, head_dim)
        self.device = device or torch.device("cuda")
        self.layer_idx = layer_idx

        self.quantizer = TurboQuantProd(
            dim=head_dim,
            bits=key_bits,
            device=self.device,
            seed=42 + layer_idx * 7,
        )

        # Quantization geometry (derived from quantizer config)
        self.mse_bits = key_bits - 1           # MSE uses b-1 bits
        self.mse_packed_d = self._packed_dim(head_dim, self.mse_bits)
        self.sign_packed_d = (head_dim + 7) // 8
        self.n_groups = head_dim // self.value_group_size

        # Write cursor
        self._write_pos: int = 0
        self._max_tokens: int = max_tokens

        # Pre-allocated flat buffers (created by preallocate())
        self._flat_mse: Optional[torch.Tensor] = None
        self._flat_signs: Optional[torch.Tensor] = None
        self._flat_norms: Optional[torch.Tensor] = None
        self._flat_res_norms: Optional[torch.Tensor] = None
        self._flat_v_data: Optional[torch.Tensor] = None
        self._flat_v_scales: Optional[torch.Tensor] = None
        self._flat_v_zeros: Optional[torch.Tensor] = None

        # Device-side N counter — updated each append, read by Triton kernel
        self._n_tensor: Optional[torch.Tensor] = None

        # Legacy chunk-list path (used when max_tokens == 0)
        self._key_chunks: list[ProdQuantized] = []
        self._value_chunks: list[ValueQuantized] = []
        self._chunk_lengths: list[int] = []

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _packed_dim(d: int, bits: int) -> int:
        if bits == 1:
            return (d + 7) // 8
        elif bits == 2:
            return (d + 3) // 4
        elif bits <= 4:
            return (d + 1) // 2
        return d

    # ------------------------------------------------------------------
    # Pre-allocation
    # ------------------------------------------------------------------

    @property
    def is_preallocated(self) -> bool:
        return self._flat_mse is not None

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def n_tensor(self) -> Optional[torch.Tensor]:
        """Device (1,) int32 tensor holding current valid token count."""
        return self._n_tensor

    def preallocate(self, max_tokens: int):
        """Pre-allocate all compressed KV buffers at a fixed capacity.

        Must be called before the first decode step.  All tensor addresses
        remain stable for the lifetime of the store — required for CUDA Graph.
        """
        H = self.num_kv_heads
        D = self.head_dim
        dev = self.device

        self._max_tokens = max_tokens
        self._write_pos = 0

        # Key buffers
        self._flat_mse = torch.zeros(
            H, max_tokens, self.mse_packed_d, device=dev, dtype=torch.uint8,
        )
        self._flat_signs = torch.zeros(
            H, max_tokens, self.sign_packed_d, device=dev, dtype=torch.uint8,
        )
        self._flat_norms = torch.zeros(H, max_tokens, device=dev, dtype=torch.float32)
        self._flat_res_norms = torch.zeros(H, max_tokens, device=dev, dtype=torch.float32)

        # Value buffers (unpacked to per-element uint8)
        self._flat_v_data = torch.zeros(
            H, max_tokens, D, device=dev, dtype=torch.uint8,
        )
        self._flat_v_scales = torch.zeros(
            H, max_tokens, self.n_groups, device=dev, dtype=torch.float32,
        )
        self._flat_v_zeros = torch.zeros(
            H, max_tokens, self.n_groups, device=dev, dtype=torch.float32,
        )

        # Device-side N counter
        self._n_tensor = torch.zeros(1, device=dev, dtype=torch.int32)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    @property
    def num_tokens(self) -> int:
        return self._write_pos + sum(self._chunk_lengths)

    @property
    def num_chunks(self) -> int:
        return len(self._chunk_lengths)

    def append_chunk(self, key: torch.Tensor, value: torch.Tensor):
        """Quantize and store a chunk of KV pairs.

        key/value: (chunk_len, num_kv_heads, head_dim)
        """
        chunk_len = key.shape[0]

        if self.is_preallocated:
            self._append_chunk_preallocated(key, value, chunk_len)
        else:
            self._append_chunk_legacy(key, value, chunk_len)

    def _append_chunk_preallocated(
        self, key: torch.Tensor, value: torch.Tensor, chunk_len: int,
    ):
        pos = self._write_pos
        H = self.num_kv_heads

        # Quantize
        k = key.transpose(0, 1).unsqueeze(0)  # (1, H, T, D)
        v = value.transpose(0, 1).unsqueeze(0)

        key_q = self.quantizer.quantize(k)
        val_q = quantize_values(v, bits=self.value_bits, group_size=self.value_group_size)

        # Flatten batch dim: (1, H, T, ...) -> (H, T, ...)
        mse = key_q.mse_indices.reshape(H, chunk_len, -1)
        signs = key_q.qjl_signs.reshape(H, chunk_len, -1)
        norms = key_q.norms.reshape(H, chunk_len)
        res_norms = key_q.residual_norms.reshape(H, chunk_len)

        # Unpack values to per-element uint8
        v_data = val_q.data
        v_scales = val_q.scales
        v_zeros_arr = val_q.zeros
        v_bits = val_q.bits if len(val_q) > 3 else self.value_bits

        if v_bits == 2 and v_data.shape[-1] != self.head_dim:
            v_data = unpack_values(val_q)
        elif v_bits == 4 and v_data.shape[-1] != self.head_dim:
            v_data = unpack_values(val_q)

        v_data = v_data.reshape(H, chunk_len, -1)
        v_scales = v_scales.reshape(H, chunk_len, -1)
        v_zeros_arr = v_zeros_arr.reshape(H, chunk_len, -1)

        # Write into pre-allocated buffer
        self._flat_mse[:, pos:pos + chunk_len].copy_(mse.contiguous())
        self._flat_signs[:, pos:pos + chunk_len].copy_(signs.contiguous())
        self._flat_norms[:, pos:pos + chunk_len].copy_(norms.contiguous())
        self._flat_res_norms[:, pos:pos + chunk_len].copy_(res_norms.contiguous())
        self._flat_v_data[:, pos:pos + chunk_len].copy_(v_data.contiguous())
        self._flat_v_scales[:, pos:pos + chunk_len].copy_(v_scales.contiguous())
        self._flat_v_zeros[:, pos:pos + chunk_len].copy_(v_zeros_arr.contiguous())

        self._write_pos += chunk_len
        # Update device-side N counter
        self._n_tensor.fill_(self._write_pos)

    def _append_chunk_legacy(self, key: torch.Tensor, value: torch.Tensor, chunk_len: int):
        """Legacy path: growing chunk list (not CUDA-Graph compatible)."""
        k = key.transpose(0, 1).unsqueeze(0)
        v = value.transpose(0, 1).unsqueeze(0)

        key_q = self.quantizer.quantize(k)
        val_q = quantize_values(v, bits=self.value_bits, group_size=self.value_group_size)

        self._key_chunks.append(key_q)
        self._value_chunks.append(val_q)
        self._chunk_lengths.append(chunk_len)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_flat_cache(self) -> Optional[FlatCache]:
        """Return flat cache view.

        For pre-allocated stores: returns the full fixed-address tensors.
        The actual valid count is in ``self.n_tensor`` (device int32).
        For legacy stores: concatenates chunks.
        """
        if self.is_preallocated:
            if self._write_pos == 0:
                return None
            return FlatCache(
                prod_q=ProdQuantized(
                    mse_indices=self._flat_mse,
                    qjl_signs=self._flat_signs,
                    norms=self._flat_norms,
                    residual_norms=self._flat_res_norms,
                    mse_bits=self.mse_bits,
                ),
                value_q=ValueQuantized(
                    data=self._flat_v_data,
                    scales=self._flat_v_scales,
                    zeros=self._flat_v_zeros,
                    bits=self.value_bits,
                ),
                num_tokens=self._write_pos,
            )

        # Legacy path
        if not self._key_chunks:
            return None

        if len(self._key_chunks) == 1:
            kq = _flatten_prod_q(self._key_chunks[0])
            vq = _flatten_value_q(self._value_chunks[0])
        else:
            kq = _concat_prod_q([_flatten_prod_q(c) for c in self._key_chunks])
            vq = _concat_value_q([_flatten_value_q(c) for c in self._value_chunks])

        return FlatCache(prod_q=kq, value_q=vq, num_tokens=self.num_tokens)

    def memory_bytes(self) -> int:
        total = 0
        if self.is_preallocated:
            total += self._flat_mse.nelement()
            total += self._flat_signs.nelement()
            total += self._flat_norms.nelement() * 4
            total += self._flat_res_norms.nelement() * 4
            total += self._flat_v_data.nelement()
            total += self._flat_v_scales.nelement() * 4
            total += self._flat_v_zeros.nelement() * 4
        else:
            for kq in self._key_chunks:
                total += kq.mse_indices.nelement()
                total += kq.qjl_signs.nelement()
                total += kq.residual_norms.nelement() * 2
                total += kq.norms.nelement() * 2
            for vq in self._value_chunks:
                total += vq.data.nelement()
                total += vq.scales.nelement() * 2
                total += vq.zeros.nelement() * 2
        return total

    def reset(self):
        self._write_pos = 0
        if self._n_tensor is not None:
            self._n_tensor.fill_(0)
        self._key_chunks.clear()
        self._value_chunks.clear()
        self._chunk_lengths.clear()


# ===================================================================
# Legacy helpers (for non-preallocated path)
# ===================================================================

def _flatten_prod_q(pq: ProdQuantized) -> ProdQuantized:
    return ProdQuantized(
        mse_indices=pq.mse_indices.reshape(-1, pq.mse_indices.shape[-2], pq.mse_indices.shape[-1]).contiguous(),
        qjl_signs=pq.qjl_signs.reshape(-1, pq.qjl_signs.shape[-2], pq.qjl_signs.shape[-1]).contiguous(),
        residual_norms=pq.residual_norms.reshape(-1, pq.residual_norms.shape[-1]).contiguous(),
        norms=pq.norms.reshape(-1, pq.norms.shape[-1]).contiguous(),
        mse_bits=pq.mse_bits,
    )


def _flatten_value_q(vq: ValueQuantized) -> ValueQuantized:
    v_bits = vq.bits if len(vq) > 3 else 2
    return ValueQuantized(
        data=vq.data.reshape(-1, vq.data.shape[-2], vq.data.shape[-1]).contiguous(),
        scales=vq.scales.reshape(-1, vq.scales.shape[-2], vq.scales.shape[-1]).contiguous(),
        zeros=vq.zeros.reshape(-1, vq.zeros.shape[-2], vq.zeros.shape[-1]).contiguous(),
        bits=v_bits,
    )


def _concat_prod_q(chunks: list[ProdQuantized]) -> ProdQuantized:
    return ProdQuantized(
        mse_indices=torch.cat([c.mse_indices for c in chunks], dim=-2),
        qjl_signs=torch.cat([c.qjl_signs for c in chunks], dim=-2),
        residual_norms=torch.cat([c.residual_norms for c in chunks], dim=-1),
        norms=torch.cat([c.norms for c in chunks], dim=-1),
        mse_bits=chunks[0].mse_bits,
    )


def _concat_value_q(chunks: list[ValueQuantized]) -> ValueQuantized:
    v_bits = chunks[0].bits if len(chunks[0]) > 3 else 2
    return ValueQuantized(
        data=torch.cat([c.data for c in chunks], dim=-2),
        scales=torch.cat([c.scales for c in chunks], dim=-2),
        zeros=torch.cat([c.zeros for c in chunks], dim=-2),
        bits=v_bits,
    )
