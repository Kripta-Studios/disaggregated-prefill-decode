"""Disaggregated prefill/decode: a small real system plus a cluster simulator.

Sub-modules
-----------
kvcodec    -- turn a transformers KV cache into one contiguous byte buffer and back
transport  -- the KV transport abstraction (TCP + shared-memory backends)
engine     -- thin greedy prefill / decode-step wrapper around a HF causal LM
cluster    -- prefill worker, decode worker, colocated worker and the router
sim        -- roofline cost model + discrete-event simulator of a heterogeneous cluster
"""

__all__ = ["kvcodec", "transport", "engine", "cluster", "sim"]
