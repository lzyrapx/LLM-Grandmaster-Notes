# LLM Grandmaster Notes

📚The path to LLM mastery is paved with broken embeddings and resurrected gradients.

- base
  + [ ] transformer
  + [x] vit transformer
  + [x] lm head
  + [ ] kv cache
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
  + [ ] grouped-query attention (GQA)
  + [ ] multi-query attention (MQA)
  + [ ] multi-head latent attention (MLA)
  + [ ] multi-token attention (MTA)
  + [x] sage attention 1
  + [x] sage attention 2
  + [x] sage attention 2++
  + [x] sage attention 3
  + [ ] paged attention
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
  + [ ] sparse
  + [ ] quantization
  + [ ] allocator
  + [ ] window
  + [ ] share
- norm
  + [ ] Layer Norm
  + [ ] RMS Norm
  + [ ] Batch Norm
- position embedding
  + [ ] RoPE
  + [ ] AliBi
  + [ ] 2D RoPE
  + [ ] 3D RoPE
  + [ ] NTK-Award RoPE
  + [ ] Yarn
- quantization
  + [ ] smooth quant
  + [ ] AWQ
  + [ ] KIVI
  + [ ] GPTQ
- design
  + [ ] chunked prefill
  + [ ] continous batching
  + [ ] speculative decoding
    + [ ] Medusa
    + [ ] Lookahead decoding
    + [ ] NGram
    + [ ] OSD
    + [ ] Eagle 1,2,3
  + [ ] sliding window
  + [ ] multi-token prediction (MTP)
- reinforcement learning
  + [ ] PPO
  + [ ] GRPO
  + [ ] DAPO
  + [ ] GPG
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
  + [ ] wgmma

