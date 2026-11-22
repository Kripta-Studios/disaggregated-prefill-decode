# State of the art: prefill/decode disaggregation (as of late 2026)

These are the papers, systems and docs I read while building this repo. I only list sources
I actually opened or saw in search results. Numbers are the authors' own claims, quoted as
they report them. I did not reproduce them.

## 1. The core papers (2023-2024)

| Work | Venue / id | Key idea | Reported result |
|---|---|---|---|
| **Splitwise**: Patel et al. (UW + Microsoft) | ISCA 2024, [arXiv:2311.18677](https://arxiv.org/abs/2311.18677) | Prompt computation is compute-intensive and token generation is memory-intensive, with different latency, throughput, memory and power profiles. Run them on separate machines, possibly with different hardware, and move the KV state over the cluster back-plane. | Up to 1.4x higher throughput at 20% lower cost, or 2.35x more throughput under the same power and cost budget. |
| **DistServe**: Zhong et al. (PKU + UCSD) | OSDI 2024, [arXiv:2401.09670](https://arxiv.org/abs/2401.09670) | Optimizes **goodput**: the request rate per GPU that meets *both* a TTFT SLO and a TPOT SLO. Prefill and decode go to different GPUs, which removes interference. Each phase gets its own parallelism, and placement depends on cluster bandwidth. | 7.4x more requests, or 12.6x tighter SLO, vs. the state of the art at the time. |
| **TetriInfer** ("Inference without Interference"): Hu et al. | [arXiv:2401.11181](https://arxiv.org/abs/2401.11181) | Disaggregated prefill/decode instances that can flip roles. Prompts are split into fixed-size chunks, and a length predictor drives decode scheduling. Built on vLLM, with a C++ network stack for the KV transfer. | Measured up to 10x slowdown from mixing prefills and 5x from mixing prefill with decode. Gains are marginal for heavy-prefill + heavy-decode mixes. |
| **Mooncake**: Qin et al. (Moonshot AI / Tsinghua) | [arXiv:2407.00079](https://arxiv.org/abs/2407.00079) (v4 Sep 2025; ACM ToS version) | A *KVCache-centric* design that serves Kimi. Separate prefill and decode clusters, plus a disaggregated KV cache pool on the cluster's spare CPU DRAM and SSD. A global scheduler ("Conductor") picks a prefill/decode pair per request under TTFT/TBT SLOs. | Up to +525% throughput in simulated scenarios and +75% more requests on real workloads (arXiv). The journal version reports +115% / +107% on A800 / H800 clusters. An open trace is published. |
| **Sarathi-Serve**: Agrawal et al. (Microsoft Research India) | OSDI 2024, [arXiv:2403.02310](https://arxiv.org/abs/2403.02310) | The *colocated* answer to interference. **Chunked prefill** plus **stall-free batching** puts a token-budgeted prompt chunk in every decode iteration. This bounds the decode stall and costs a little TTFT. | 2.6x serving capacity for Mistral-7B on one A100 and up to 3.7x for Yi-34B on two A100s vs. vLLM, under tail-latency SLOs. |

The surveys I found that cover the topic are *Taming the Titans: A Survey of Efficient LLM
Inference Serving* ([arXiv:2504.19720](https://arxiv.org/abs/2504.19720)) and *From
Attention to Disaggregation: Tracing the Evolution of LLM Inference*
([arXiv:2511.07422](https://arxiv.org/abs/2511.07422)). A 2026 follow-up line looks at
multi-turn prefills, for example *Not All Prefills Are Equal: PPD Disaggregation for
Multi-turn LLM Serving* ([arXiv:2603.13358](https://arxiv.org/abs/2603.13358)).

## 2. Production systems

- **NVIDIA Dynamo** ([docs](https://docs.nvidia.com/dynamo/)) is a datacenter-scale
  orchestration layer over vLLM, SGLang and TensorRT-LLM. It has disaggregated prefill and
  decode workers and KV-aware routing. With TensorRT-LLM, KV moves through **NIXL**. The
  docs say UCX is the default NIXL backend, and LIBFABRIC is added in v1.1. They also
  recommend UCX/SRD for AWS EFA
  ([KV cache transfer page](https://docs.nvidia.com/dynamo/v1.1.0/additional-resources/tensor-rt-llm-details/kv-cache-transfer)).
- **NIXL, the NVIDIA Inference Xfer Library** ([github.com/ai-dynamo/nixl](https://github.com/ai-dynamo/nixl),
  [overview](https://docs.nvidia.com/nixl/getting-started/overview)) is a point-to-point
  transfer library. One API covers GPU memory, CPU memory and storage, with plug-in
  backends (UCX, GDS, ...) and non-blocking, non-contiguous transfers. Dynamo,
  TensorRT-LLM, vLLM, SGLang, Ray and LMCache use it. It ships a benchmark, `nixlbench`.
- **vLLM** has an experimental disaggregated-prefill feature with KV *connectors*
  ([docs](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)). The `NixlConnector`
  does fully async send/recv. The docs say plainly that **disaggregated prefill does not
  improve throughput**: the benefit is control over tail latency (TTFT and ITL tuned
  separately).
- **llm-d** ([Red Hat intro](https://developers.redhat.com/articles/2025/08/19/getting-started-llm-d-distributed-ai-inference),
  [AWS post](https://aws.amazon.com/blogs/machine-learning/introducing-disaggregated-inference-on-aws-powered-by-llm-d))
  is a Kubernetes-native stack: vLLM, the Gateway API Inference Extension, KV-cache-aware
  routing and P/D disaggregation. A routing sidecar tells vLLM to do point-to-point KV
  transfers, and NIXL moves the bytes
  ([NVIDIA blog](https://developer.nvidia.com/blog/nvidia-dynamo-accelerates-llm-d-community-initiatives-for-advancing-large-scale-distributed-inference)).
  The reference P/D example targets large models (Llama-70B class) on H200s.
- **SGLang** offers PD disaggregation with Mooncake (the default) or NIXL as the RDMA
  transfer engine. The decode server pre-allocates KV, the prefill server computes it and
  writes it across. LMSYS reports DeepSeek-V3/R1 on **96 H100s (12 nodes)** with PD
  disaggregation and large-scale expert parallelism: about 52.3k input and 22.3k output
  tokens/s per node for 2k-token inputs
  ([blog, May 2025](https://lmsys.org/blog/2025-05-05-large-scale-ep/)). A follow-up runs
  Kimi K2 on 128 H200s ([blog, Jul 2025](https://lmsys.org/blog/2025-07-20-k2-large-scale-ep)).
  AMD documents the same pattern on MI300X
  ([ROCm blog](https://rocm.blogs.amd.com/software-tools-optimization/disaggregation/README.html)).

## 3. What this means for the design in this repo

1. **The metric is goodput under two SLOs** (DistServe), not raw throughput. The vLLM docs
   make the same point from the other side: disaggregation exists to decouple TTFT from ITL.
2. **Chunked prefill is the baseline to beat**, not naive colocation, because Sarathi-Serve
   removes much of the interference without a network. Our simulator models both colocated
   policies.
3. **The receiver owns the destination memory**, and the transfer is write-then-notify
   (SGLang, NIXL). Our `ShmTransport` copies that protocol: a decode-owned arena, slot
   credits, a notification queue. An RDMA/NIXL backend would keep the same interface.
4. **Heterogeneity is about the right roof per phase** (Splitwise). Prefill wants FLOPs.
   Decode wants HBM bandwidth and capacity. A cheaper GPU only wins the decode role when
   its *bandwidth per dollar* is competitive.

## 4. Hardware numbers used by the simulator (datasheet, dense BF16)

| GPU | BF16 dense | HBM BW | Memory | Source |
|---|---|---|---|---|
| H100 SXM | 989 TFLOPS (1,979 with sparsity) | 3.35 TB/s | 80 GB HBM3 | [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/), [Hopper in-depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) |
| L40S | 362 TFLOPS (733 with sparsity) | 864 GB/s | 48 GB GDDR6 | [NVIDIA L40S](https://www.nvidia.com/en-us/data-center/l40s/), [Lenovo Press LP1812](https://lenovopress.lenovo.com/lp1812.pdf) |

Network bandwidths used in the sweeps are nominal line rates (10 GbE = 1.25 GB/s,
100 Gb RoCE = 12.5 GB/s, 400 Gb NDR InfiniBand = 50 GB/s) and an NVLink-class 450 GB/s
reference point. They are not measurements.
