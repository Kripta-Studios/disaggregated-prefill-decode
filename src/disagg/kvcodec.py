"""Serialize a KV cache into ONE contiguous buffer (and back).

Why one buffer?  A transformers ``DynamicCache`` is a list of ``num_layers`` (K, V) pairs,
i.e. 2 * L separate tensors.  Shipping them one by one means 2 * L small transfers, each
paying a fixed per-message cost.  Production systems solve this in two ways:

* vLLM / SGLang + NIXL or Mooncake register the paged KV pool once and issue a *list* of
  (address, length) descriptors in a single RDMA operation -- no copy at all.
* Simpler systems gather all layers into a staging buffer and send that.

We do the second: ``torch.stack`` gathers every layer into a single GPU tensor of shape
``[L, 2, B, H_kv, T, D]`` (one kernel), then one device->host copy lands it in pinned
memory.  The receiver does the reverse with one host->device copy.  The two PCIe hops are
exactly what GPUDirect RDMA removes in production (NIC reads/writes GPU memory directly).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

# torch dtypes we may ship, by name (JSON-friendly metadata).
_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class PackTiming:
    gather_s: float  # GPU-side stack of all layers into one tensor
    d2h_s: float  # device -> pinned host copy


def kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    """Bytes of KV cache one token occupies: K and V, every layer, every KV head."""
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def cache_to_layers(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract per-layer (K, V) tensors from a transformers Cache (v5 ``.layers`` API)."""
    return [(layer.keys, layer.values) for layer in cache.layers]


def pack_kv(layers: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[dict, torch.Tensor, PackTiming]:
    """Gather all layers into one pinned uint8 host tensor.

    Returns (meta, payload, timing).  ``meta`` is everything the receiver needs to rebuild
    the tensors: dtype and the 6-D shape.  ``payload`` is a flat uint8 CPU tensor.
    """
    k0 = layers[0][0]
    sync = torch.cuda.synchronize if k0.is_cuda else (lambda: None)
    sync()
    t0 = time.perf_counter()
    stacked = torch.stack([torch.stack((k, v)) for k, v in layers])  # [L, 2, B, H, T, D]
    sync()
    t1 = time.perf_counter()
    flat = stacked.view(-1).view(torch.uint8)  # reinterpret bf16 bytes, no copy
    host = torch.empty(flat.numel(), dtype=torch.uint8, pin_memory=k0.is_cuda)
    host.copy_(flat)  # one DMA over PCIe (synchronous for pinned dst on the default stream)
    sync()
    t2 = time.perf_counter()
    dtype_name = next(n for n, d in _DTYPES.items() if d == k0.dtype)
    meta = {"dtype": dtype_name, "shape": list(stacked.shape), "nbytes": host.numel()}
    return meta, host, PackTiming(gather_s=t1 - t0, d2h_s=t2 - t1)


def unpack_kv(meta: dict, payload, device: str | torch.device = "cpu") -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Rebuild per-layer (K, V) tensors on ``device`` from a received byte buffer.

    ``payload`` may be any buffer-protocol object (bytearray, shared-memory slice...).  We copy
    it to ``device`` immediately, so the caller may recycle the buffer as soon as this returns.
    """
    raw = torch.frombuffer(payload, dtype=torch.uint8, count=meta["nbytes"])
    # copy=True: never alias the transport buffer (it is recycled right after this call)
    t = raw.to(device, copy=True).view(_DTYPES[meta["dtype"]]).view(meta["shape"])
    return [(t[i, 0], t[i, 1]) for i in range(t.shape[0])]


def layers_to_cache(layers: list[tuple[torch.Tensor, torch.Tensor]], config=None):
    """Wrap (K, V) tensors into a fresh ``DynamicCache`` the model can continue decoding from."""
    from transformers import DynamicCache

    cache = DynamicCache(config=config) if config is not None else DynamicCache()
    for i, (k, v) in enumerate(layers):
        cache.update(k, v, i)
    return cache
