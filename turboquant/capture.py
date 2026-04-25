"""
TurboQuant capture module — bulk ingestion and ring-buffer management.

Handles the write path:
  - Bulk capture from paged KV cache or raw tensors (prefill)
  - Append decode tokens into a small exact ring buffer
  - Flush ring buffer to compressed store only when full or at phase boundaries

Design rule: no per-token quantization on the hot decode path.
"""

from __future__ import annotations

import torch
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from turboquant.store import CompressedKVStore


class RingBuffer:
    """Fixed-size ring buffer for recent exact KV tokens.

    Stores the most recent ``capacity`` tokens in bf16/fp16.
    When full, the oldest chunk is returned for compression.

    CUDA-Graph mode:
      When ``preallocate_graph_buffers()`` is called, the ring buffer uses
      device-side int32 tensors for position and count.  All write/read
      operations are pure CUDA ops (index_copy_, add_, remainder_, etc.)
      so they can be captured and replayed by CUDA Graph without any
      Python involvement.
    """

    __slots__ = (
        "capacity",
        "num_kv_heads",
        "head_dim",
        "device",
        "dtype",
        "_k",
        "_v",
        "_pos",
        "_total_written",
        # CUDA-Graph device tensors (set by preallocate_graph_buffers)
        "_pos_tensor",
        "_count_tensor",
        "_capacity_tensor",
        "_arange_buf",
        "_cpu_decode_steps",
        "_graph_mode",
    )

    def __init__(
        self,
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.capacity = capacity
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self._k = torch.zeros(
            capacity, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        self._v = torch.zeros(
            capacity, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        self._pos = 0
        self._total_written = 0

        # CUDA-Graph device tensors (initialised lazily)
        self._pos_tensor = None
        self._count_tensor = None
        self._capacity_tensor = None
        self._arange_buf = None
        self._cpu_decode_steps = 0
        self._graph_mode = False

    @property
    def size(self) -> int:
        return self._pos

    @property
    def is_full(self) -> bool:
        return self._pos >= self.capacity

    @property
    def total_written(self) -> int:
        return self._total_written

    def write(
        self, key: torch.Tensor, value: torch.Tensor, num_tokens: int
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Append tokens. Returns (overflow_k, overflow_v) if buffer overflows, else None.

        key/value shapes: (num_tokens, num_kv_heads, head_dim)
        """
        space = self.capacity - self._pos
        overflow_k_parts = []
        overflow_v_parts = []

        offset = 0
        remaining = num_tokens

        while remaining > 0:
            space = self.capacity - self._pos
            if space <= 0:
                # Buffer is full — drain it
                overflow_k_parts.append(self._k[: self._pos].clone())
                overflow_v_parts.append(self._v[: self._pos].clone())
                self._pos = 0
                space = self.capacity

            n = min(remaining, space)
            self._k[self._pos : self._pos + n] = key[offset : offset + n]
            self._v[self._pos : self._pos + n] = value[offset : offset + n]
            self._pos += n
            offset += n
            remaining -= n

        self._total_written += num_tokens

        if overflow_k_parts:
            return (
                torch.cat(overflow_k_parts, dim=0),
                torch.cat(overflow_v_parts, dim=0),
            )
        return None

    def drain(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return all buffered tokens and reset. Returns None if empty."""
        if self._pos == 0:
            return None
        k = self._k[: self._pos].clone()
        v = self._v[: self._pos].clone()
        self._pos = 0
        return k, v

    def peek(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Read current buffer contents without draining."""
        if self._pos == 0:
            return None
        return self._k[: self._pos], self._v[: self._pos]

    def reset(self):
        self._pos = 0
        self._total_written = 0
        self._cpu_decode_steps = 0
        if self._graph_mode:
            self._pos_tensor.fill_(0)
            self._count_tensor.fill_(0)

    # ------------------------------------------------------------------
    # CUDA-Graph-compatible methods
    # ------------------------------------------------------------------

    def preallocate_graph_buffers(self):
        """Initialise device tensors for CUDA-Graph-compatible operation.

        Call once during model initialisation, before any CUDA Graph capture.
        """
        self._pos_tensor = torch.zeros(1, device=self.device, dtype=torch.int64)
        self._count_tensor = torch.zeros(1, device=self.device, dtype=torch.int64)
        self._capacity_tensor = torch.tensor(
            [self.capacity], device=self.device, dtype=torch.int64,
        )
        self._arange_buf = torch.arange(
            self.capacity, device=self.device, dtype=torch.int64,
        )
        self._graph_mode = True

    @property
    def graph_ready(self) -> bool:
        return self._graph_mode

    def write_graph(self, key: torch.Tensor, value: torch.Tensor):
        """CUDA-Graph-compatible write for a single decode token.

        All operations are in-place CUDA ops on fixed-address tensors:
          1. index_copy_ writes K/V at current device-side position
          2. Position increments with wrap-around (remainder_)
          3. Count increments and clamps to capacity

        key/value: (1, num_kv_heads, head_dim) or (num_kv_heads, head_dim)
        """
        if key.dim() == 2:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

        # Write K/V at current position (cast to buffer dtype if needed)
        self._k.index_copy_(0, self._pos_tensor, key.to(self._k.dtype))
        self._v.index_copy_(0, self._pos_tensor, value.to(self._v.dtype))

        # Advance position (wrap around at capacity)
        self._pos_tensor.add_(1)
        self._pos_tensor.remainder_(self._capacity_tensor)

        # Advance and clamp count (minimum_ not available in all PyTorch versions)
        self._count_tensor.add_(1)
        torch.minimum(self._count_tensor, self._capacity_tensor, out=self._count_tensor)

    def peek_full(self):
        """Return full ring buffer + device count for CUDA-Graph-compatible read.

        Returns:
            (ring_k, ring_v, count_tensor, arange_buf, capacity_tensor)
            - ring_k: (capacity, H, D) — full buffer
            - ring_v: (capacity, H, D) — full buffer
            - count_tensor: (1,) int64 — number of valid entries
            - arange_buf: (capacity,) int64 — for masking
            - capacity_tensor: (1,) int64 — ring buffer capacity
        """
        return self._k, self._v, self._count_tensor, self._arange_buf, self._capacity_tensor

    def is_full_for_graph(self) -> bool:
        """CPU-side check: has the ring buffer filled up since last compression?"""
        return self._cpu_decode_steps >= self.capacity

    def reset_for_graph(self):
        """Reset device tensors after compression (called between graph replays)."""
        self._pos_tensor.fill_(0)
        self._count_tensor.fill_(0)
        self._cpu_decode_steps = 0


class KVCaptureEngine:
    """Orchestrates capture of KV pairs into a CompressedKVStore.

    Sits between the vLLM attention backend and the compressed store.
    Manages the ring buffer and decides when to flush to the store.
    """

    def __init__(
        self,
        store: "CompressedKVStore",
        ring_capacity: int = 128,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.store = store
        self.ring = RingBuffer(
            capacity=ring_capacity,
            num_kv_heads=store.num_kv_heads,
            head_dim=store.head_dim,
            device=device or store.device,
            dtype=dtype,
        )
        self._prefill_done = False
        self._was_decoding = False

    @property
    def total_compressed_tokens(self) -> int:
        return self.store.num_tokens

    @property
    def total_buffered_tokens(self) -> int:
        return self.ring.size

    @property
    def total_tokens(self) -> int:
        return self.total_compressed_tokens + self.total_buffered_tokens

    def ingest_prefill(self, key: torch.Tensor, value: torch.Tensor, num_tokens: int):
        """Bulk-capture prefill KV into the store (bypasses ring buffer).

        key/value: (num_tokens, num_kv_heads, head_dim)
        """
        if self._was_decoding:
            # Previous request's decode state is still present — new request starting.
            self.reset()

        if num_tokens <= self.ring.capacity:
            self.ring.write(key[:num_tokens], value[:num_tokens], num_tokens)
        else:
            n_compress = num_tokens - self.ring.capacity
            self.store.append_chunk(key[:n_compress], value[:n_compress])
            self.ring.write(
                key[n_compress:num_tokens],
                value[n_compress:num_tokens],
                self.ring.capacity,
            )
        self._prefill_done = True

    def ingest_prefill_from_paged_cache(
        self,
        kv_cache_tensor: torch.Tensor,
        num_tokens: int,
        block_table: torch.Tensor,
        block_size: int,
    ):
        """Bulk-capture prefill by reading from vLLM's paged KV cache tensor.

        kv_cache_tensor: (2, num_blocks, block_size, num_kv_heads, head_dim)
        block_table: (num_blocks_used,) int — maps logical block idx -> physical
        """
        num_blocks_needed = (num_tokens + block_size - 1) // block_size
        physical_blocks = block_table[:num_blocks_needed]

        keys_list = []
        vals_list = []
        collected = 0

        for i, phys_idx in enumerate(physical_blocks):
            start = 0
            end = min(block_size, num_tokens - collected)
            k_block = kv_cache_tensor[0, phys_idx, start:end]  # (end, heads, dim)
            v_block = kv_cache_tensor[1, phys_idx, start:end]
            keys_list.append(k_block)
            vals_list.append(v_block)
            collected += end

        all_k = torch.cat(keys_list, dim=0)  # (num_tokens, heads, dim)
        all_v = torch.cat(vals_list, dim=0)
        self.ingest_prefill(all_k, all_v, num_tokens)

    def ingest_decode(self, key: torch.Tensor, value: torch.Tensor, num_tokens: int):
        """Append decode tokens. Cheap: just writes to ring buffer.

        Overflow is automatically flushed to the compressed store.
        key/value: (num_tokens, num_kv_heads, head_dim)
        """
        self._was_decoding = True
        if self.ring._graph_mode and num_tokens == 1:
            # CUDA-Graph-compatible path: single-token device-tensor write
            self.ring.write_graph(key[:1], value[:1])
        else:
            overflow = self.ring.write(key[:num_tokens], value[:num_tokens], num_tokens)
            if overflow is not None:
                k_over, v_over = overflow
                self.store.append_chunk(k_over, v_over)

    def prepare_for_decode(self):
        """Compress prefill ring buffer data before entering decode / graph mode.

        Called once at the prefill-to-decode transition.  Drains the ring
        buffer into the compressed store and resets device-side counters.
        """
        data = self.ring.drain()
        if data is not None:
            k, v = data
            self.store.append_chunk(k, v)
        if self.ring._graph_mode:
            self.ring.reset_for_graph()

    def check_overflow_and_compress(self):
        """Check if the ring buffer has filled up and compress if needed.

        Called between CUDA Graph replays (from the model runner hook).
        Uses CPU-side tracking — no CUDA sync required.
        """
        if not self.ring._graph_mode:
            return
        if not self.ring.is_full_for_graph():
            return

        # Clone ring buffer and compress into store
        k = self.ring._k.clone()
        v = self.ring._v.clone()
        self.store.append_chunk(k, v)
        self.ring.reset_for_graph()

    def flush(self):
        """Force-flush ring buffer to compressed store."""
        data = self.ring.drain()
        if data is not None:
            k, v = data
            self.store.append_chunk(k, v)

    def reset(self):
        self.ring.reset()
        self.store.reset()
        self._prefill_done = False
        self._was_decoding = False
