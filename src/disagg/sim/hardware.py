"""Roofline cost model: how long does one engine iteration take on a given GPU?

An iteration processes some *new tokens* (prompt tokens being prefilled and/or one token per
running decode).  Its time is bounded below by two roofs:

    t_compute = FLOPs / (peak_FLOPs * MFU)          FLOPs ~= 2 * P_matmul * new_tokens + attention
    t_memory  = bytes / (peak_BW * MBU)             bytes ~= weights + KV read + KV written
    t_iter    = max(t_compute, t_memory) + fixed_overhead

Prefill of T tokens reuses each weight byte T times -> compute-bound once T exceeds the
"ridge point" peak_FLOPs / peak_BW (~295 FLOP/B on an H100, i.e. a few hundred tokens).
Decode processes one token per sequence -> with batch B, intensity ~= B FLOP/B, so it stays
memory-bound until B reaches the ridge, and KV reads grow the byte count further.

Peak numbers are public datasheet values (dense, no sparsity); MFU/MBU/overhead are
*assumptions* of this model (see README), not measurements.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    peak_flops: float  # dense BF16 tensor FLOP/s
    hbm_bw: float  # bytes/s
    hbm_bytes: float
    mfu: float = 0.55  # achievable fraction of peak FLOPs on large GEMMs (assumption)
    mbu: float = 0.75  # achievable fraction of peak bandwidth when streaming weights/KV (assumption)
    overhead_s: float = 1.5e-3  # per-iteration launch/scheduler overhead (assumption)


# Datasheet values: H100 SXM 989 TFLOPS BF16 dense, 3.35 TB/s, 80 GB;  L40S 362 TFLOPS BF16
# dense, 864 GB/s, 48 GB (see docs/SOTA.md for sources).
H100 = GPU("H100-SXM", 989e12, 3.35e12, 80e9)
L40S = GPU("L40S", 362e12, 864e9, 48e9)


@dataclass(frozen=True)
class Model:
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    tied_embeddings: bool = False
    dtype_bytes: int = 2

    @property
    def layer_params(self) -> int:
        attn = self.d_model * self.head_dim * (2 * self.n_heads + 2 * self.n_kv_heads)  # q, o, k, v
        mlp = 3 * self.d_model * self.d_ff  # gate, up, down (SwiGLU)
        return attn + mlp

    @property
    def weight_bytes(self) -> int:
        emb = self.vocab * self.d_model * (1 if self.tied_embeddings else 2)
        return (self.n_layers * self.layer_params + emb) * self.dtype_bytes

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.dtype_bytes

    def kv_capacity_tokens(self, gpu: GPU, mem_util: float = 0.9) -> int:
        return int((gpu.hbm_bytes * mem_util - self.weight_bytes) // self.kv_bytes_per_token)


LLAMA31_8B = Model("Llama-3.1-8B", 32, 4096, 32, 8, 128, 14336, 128256)
QWEN3_06B = Model("Qwen3-0.6B", 28, 1024, 16, 8, 128, 3072, 151936, tied_embeddings=True)


def iteration_time(gpu: GPU, m: Model, new_tokens: int, n_seqs: int, attn_ctx: float, kv_read_tokens: float) -> float:
    """Roofline time of one forward pass.

    new_tokens     -- tokens computed this iteration (prompt chunk tokens + 1 per decode)
    n_seqs         -- sequences in the batch (each needs one LM-head row for its last token)
    attn_ctx       -- sum over new tokens of the context length each one attends to
    kv_read_tokens -- KV-cache tokens that must be streamed from HBM (decode attention)
    """
    flops = 2 * m.n_layers * m.layer_params * new_tokens  # all GEMMs
    flops += 2 * m.vocab * m.d_model * n_seqs  # LM head, last position only
    flops += 4 * m.n_layers * m.n_heads * m.head_dim * attn_ctx  # QK^T and PV
    bytes_ = m.weight_bytes + m.kv_bytes_per_token * (kv_read_tokens + new_tokens)  # weights, KV read, KV write
    return max(flops / (gpu.peak_flops * gpu.mfu), bytes_ / (gpu.hbm_bw * gpu.mbu)) + gpu.overhead_s


def prefill_time(gpu: GPU, m: Model, prompt_lens: list[int]) -> float:
    """A batch of whole prompts. Causal attention: token i attends to i tokens -> sum ~= T^2/2."""
    return iteration_time(gpu, m, sum(prompt_lens), len(prompt_lens), sum(t * t / 2 for t in prompt_lens), 0)


def decode_time(gpu: GPU, m: Model, batch: int, ctx_sum: float) -> float:
    """One decode step for ``batch`` sequences whose contexts sum to ``ctx_sum`` tokens."""
    return iteration_time(gpu, m, batch, batch, ctx_sum, ctx_sum)
