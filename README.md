# LLM Grandmaster Notes

📚The path to LLM mastery is paved with broken embeddings and resurrected gradients.

- base
  + [x] transformer
  + [x] vit transformer
  + [x] lm head
  + [x] kv cache
  - GPU Architecture
    + [ ] SM80
    + [ ] SM90
    + [ ] SM100
    + [ ] SM120
    + [ ] Memory Hierarchy (HBM, L2 Cache, Shared Memory/SMEM, Register File)
    + [ ] Warp Specialization (Producer-Consumer Model used in FA3)
    + [ ] Distributed Shared Memory (DSM) / Thread Block Clusters
- activation & mlp
  + [ ] SwiGLU / GeGLU / ReGLU
  + [ ] LLaMA MLP (Gated Linear Units)
- attention
  + [x] self attention
  + [x] online attention
  + [x] flash attention
  + [x] flash attention 2
  + [x] flash attention 3
  + [x] flash decoding
  + [x] flash decoding++
  + [x] scaled dot-product attention (SDPA)
  + [x] multi-head self-attention (MHSA)
  + [x] multi-head attention (MHA)
  + [x] grouped-query attention (GQA)
  + [x] multi-query attention (MQA)
  + [ ] multi-head latent attention (MLA)
  + [ ] multi-token attention (MTA)
  + [x] sage attention 1
  + [x] sage attention 2
  + [x] sage attention 2++
  + [x] sage attention 3
  + [x] paged attention
  + [ ] ring attention
  + [ ] ring flash attention
  + [ ] linear attention
  + [ ] lightning attention
  + [ ] native sparse attention (NSA)
  + [ ] grouped latent attention (GLA)
  + [ ] grouped-tied attention (GTA)
- softmax
  + [x] softmax
  + [x] safe softmax
  + [x] online softmax
- kv cache optimization
  + [x] sparse
  + [x] quantization
  + [x] allocator
  + [x] window
  + [x] share
- norm
  + [x] Batch Norm
  + [x] Layer Norm
  + [x] RMS Norm
- position embedding
  + [ ] RoPE
  + [x] AliBi
  + [ ] 2D RoPE
  + [ ] 3D RoPE
  + [ ] NTK-Award RoPE
  + [ ] Yarn
- quantization
  + [ ] smooth quant
  + [ ] AWQ
  + [ ] KIVI
  + [ ] GPTQ
  + [ ] FP8 Training & Inference (E4M3 / E5M2 formats, AMAX scaling)
  + [ ] FP4 / FP6 / NF4 (QLoRA)
  + [ ] NVFP4 / MXFP4 / MXFP8 (Microscaling Formats, OCP specification)
  + [ ] KV Cache Quantization (INT4 / INT8 / FP8 KV)
  + [ ] AQLM / QuIP / QuIP# (Advanced vector quantization)
- speculative decoding
    + [ ] Medusa
    + [ ] Lookahead decoding
    + [ ] NGram
    + [ ] OSD
    + [ ] Eagle 1,2,3
    + [ ] multi-token prediction (MTP)
    + [ ] Dflash
- design
  + [ ] chunked prefill
  + [ ] continous batching
  + [ ] sliding window
  + [ ] CUDA Graph (Minimizing CPU launch overhead for short decoding steps)
  + [ ] Radix Attention / Prompt Cache (SGLang style prefix caching)
  + [ ] FlashDecoding with KV Splitting (Split-K attention for long contexts)
  + [ ] Chunked Prefill & Decode Co-run (handling prefill/decode interference)
- reinforcement learning
  + [ ] PPO
  + [ ] GRPO
  + [ ] DAPO
  + [ ] GPG
  + [ ] DPO
  + [ ] KTO
  + [ ] IPO
  + [ ] SimPO
  + [ ] Rejection Sampling Fine-Tuning (RFT)
  + [ ] Online DPO / RLAIF (RL from AI Feedback)
- gemm
  + [ ] deep gemm
  - cutlass
    + [x] cooperative and ping-pong gemm scheduler
  - cublas
- open source
  + [ ] flash mla
- ptx instructions
  + [x] mbarrier
  + [x] cp.async
  + [x] ldmatrix
  + [x] mma
  + [x] wgmma
  + [ ] cp.async.bulk / TMA (Tensor Memory Accelerator for SM90/SM100)
  + [ ] tcgen05.alloc / tcgen05.mma (Blackwell 5th Gen Tensor Core MMA)
  + [ ] TMEM (Tensor Memory) access primitives (Blackwell new memory space)
- distributed parallel
  + [ ] Tensor Parallelism (TP) (Megatron-style)
  + [ ] Pipeline Parallelism (PP) (1F1B, Interleaved 1F1B, Zero-Bubble)
  + [ ] Sequence Parallelism (SP) (Megatron SP, DeepSpeed Ulysses)
  + [ ] Context Parallelism (CP) (Ring-based long context)
  + [ ] Expert Parallelism (EP) (for MoE)
  + [ ] ZeRO (Zero Redundancy Optimizer) / FSDP (ZeRO-1, ZeRO-2, ZeRO-3)
  + [ ] Activation Checkpointing (Gradient Checkpointing)
  + [ ] Communication Primitives (AllReduce, AllGather, ReduceScatter, P2P)
  + [ ] Hardware Topology (NVLink, NVSwitch, PCIe, InfiniBand, RoCE)
- mixture of experts (moe)
  + [ ] Top-k Routing (Sparse MoE)
  + [ ] Shared Experts (DeepSeekMoE style)
  + [ ] Device-Limited Routing / Segmented Routing
  + [ ] Soft MoE / Fully-Differentiable MoE
  + [ ] Expert Capacity & Auxiliary Loss (Load Balancing)
  + [ ] Dropless MoE
  + [ ] Expert Offloading (for constrained inference)

