# KV Cache

## 介绍

KV Cache 是 LLM 推理优化的核心技术之一。在自回归（Autoregressive）生成过程中，每生成一个新的 token，都需要对之前所有 token 做注意力计算。如果不使用缓存，每次生成都需要重新计算所有之前 token 的 Key 和 Value，造成大量重复计算。

KV Cache 的核心思想：**缓存已计算过的 Key 和 Value，每次新 token 只需计算自己的 Q/K/V，然后用新 Q 与缓存的 K/V 做注意力即可。**

## 自回归推理中的注意力计算

### Prefill 阶段

输入 prompt 的所有 token 被一次性并行处理：

$$
S = QK^T, \quad P = \text{softmax}(S / \sqrt{d_k}), \quad O = PV
$$

这个计算类似于矩阵乘法，可以充分利用 GPU 并行性。

### Decode 阶段

每次只生成一个新的 token，假设当前是第 $t$ 步：

- 只有新 token $x_t$ 需要计算 $q_t, k_t, v_t$
- 将 $k_t, v_t$ 追加到 KV Cache 中
- 使用 $q_t$ 与完整的缓存 $[k_1, k_2, \ldots, k_t]$ 做点积得到注意力分数
- 使用缓存的 $[v_1, v_2, \ldots, v_t]$ 加权求和得到输出

```
Step t:
  q_t = x_t @ W_Q          # shape: (1, d_k)
  k_t = x_t @ W_K          # shape: (1, d_k)
  v_t = x_t @ W_V          # shape: (1, d_v)

  K_cache = [K_cache; k_t]  # shape: (t, d_k) — 追加新 key
  V_cache = [V_cache; v_t]  # shape: (t, d_v) — 追加新 value

  s_t = q_t @ K_cache.T     # shape: (1, t)
  a_t = softmax(s_t / sqrt(d_k))  # shape: (1, t)
  o_t = a_t @ V_cache       # shape: (1, d_v)
```

## 显存占用分析

KV Cache 的显存占用是 LLM 推理的主要瓶颈之一。以一个典型的 LLM 为例：

对于一个 $L$ 层、 $n_{kv}$ 个 KV 头、头维度 $d_h$ 的模型，处理 Batch Size 为 $b$，序列长度为 $s$ 时的 KV Cache 大小：

$$
  \text{KV Cache Size} = 2 \times b \times L \times s \times n_{kv} \times d_h \times \text{sizeof(dtype)}
  $$

其中：
- $2$ 表示 Key 和 Value 各一份
- $b$ 表示 Batch Size
- $L$ 是 Transformer 层数
- $s$ 是序列长度
- $n_{kv}$ 是 KV 头数
- $d_h$ 是每个头的维度
- `sizeof(dtype)` 是数据类型大小（FP16 = 2 bytes, FP8 = 1 byte）

### 示例：LLaMA-2-70B

- $Batc Size = 1$, $L = 80$, $n_{kv} = 8$, $d_h = 128$, dtype = FP16
- 序列长度 $s = 4096$：

$$
\text{Size} = 2 \times 1 \times 80 \times 4096 \times 8 \times 128 \times 2 = 1.25 \text{ GB}
$$

当 batch size 增大或序列长度增长时，KV Cache 显存占用急剧增加，成为推理吞吐的瓶颈。

## KV Cache 与注意力变体的关系

| 注意力类型 | KV 头数 | KV Cache 大小 | 代表模型 |
|:---:|:---:|:---:|:---:|
| MHA (Multi-Head Attention) | $n_{kv} = n_{heads}$ | 最大 | GPT-3, LLaMA-1 |
| GQA (Grouped-Query Attention) | $n_{kv} < n_{heads}$ | 中等 | LLaMA-2, Gemma 系列 |
| MQA (Multi-Query Attention) | $n_{kv} = 1$ | 最小 | PaLM, Falcon |
| MLA (Multi-Head Latent Attention) | 压缩投影 | 极小 | DeepSeek-V2/V3 |

GQA 和 MQA 通过减少 KV 头数来降低 KV Cache 显存占用，MLA 则通过低秩投影将 KV 压缩到更低维度。

## Prefill vs Decode 的计算特性

| 特性 | Prefill | Decode |
|:---:|:---:|:---:|
| 计算模式 | 矩阵乘法（GEMM） | 矩阵-向量乘法（GEMV） |
| 计算密度 | 高（算术密集型） | 低（访存密集型） |
| 瓶颈 | 计算（Compute-bound） | 访存（Memory-bound） |
| 并行度 | 高（所有 token 并行） | 低（单 token 生成） |
| KV Cache | 首次构建 | 只读 + 追加一个 token |
| 核心优化技术 | FlashAttention (减少访存，加速计算) | PagedAttention (解决显存碎片), 连续批处理 (Continuous Batching), 投机采样 (Speculative Decoding) |
