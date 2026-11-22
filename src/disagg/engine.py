"""Minimal greedy inference engine: one prefill call, then one-token decode steps.

This is intentionally *not* a batched engine: each request owns a ``DynamicCache`` and a
decode "iteration" steps every active request once.  That keeps the code short and makes
the phenomenon we care about visible: whenever a prefill runs on the same GPU stream, every
in-flight decode waits for it (prefill/decode interference).
"""

from __future__ import annotations

import torch

MODEL_ID = "Qwen/Qwen3-0.6B"


class Engine:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda"):
        from transformers import AutoModelForCausalLM

        self.device = torch.device(device)
        # bf16 + low_cpu_mem_usage + direct-to-GPU keeps host RAM near the weight size (~1.2 GB).
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=device
        ).eval()
        self.config = self.model.config

    @torch.inference_mode()
    def prefill(self, prompt_ids: list[int]):
        """Run the whole prompt in one forward pass.  Returns (first_token, cache).

        This is the compute-bound phase: T prompt tokens go through every weight matrix
        together, so each weight byte loaded from HBM is reused T times.
        """
        from transformers import DynamicCache

        ids = torch.tensor([prompt_ids], device=self.device)
        out = self.model(input_ids=ids, past_key_values=DynamicCache(config=self.config), use_cache=True, logits_to_keep=1)
        return int(out.logits[0, -1].argmax()), out.past_key_values

    @torch.inference_mode()
    def decode_step(self, cache, token: int) -> int:
        """Feed one token, append its K/V to ``cache`` in place, return the next token.

        This is the memory-bound phase: one token per weight byte loaded, plus a full read
        of this request's KV cache for attention.
        """
        ids = torch.tensor([[token]], device=self.device)
        out = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
        return int(out.logits[0, -1].argmax())  # .item() also synchronizes -> honest timing

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
