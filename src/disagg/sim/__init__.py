"""Roofline cost model (``hardware``) + discrete-event cluster simulator (``simulator``).

Everything in this sub-package is SIMULATED: no number it produces was measured on an
H100 or an L40S.
"""

from .hardware import H100, L40S, LLAMA31_8B, QWEN3_06B, GPU, Model, decode_time, prefill_time
from .simulator import Cluster, Workload, attainment, goodput, simulate

__all__ = ["GPU", "Model", "H100", "L40S", "LLAMA31_8B", "QWEN3_06B", "prefill_time", "decode_time",
           "Cluster", "Workload", "simulate", "attainment", "goodput"]
