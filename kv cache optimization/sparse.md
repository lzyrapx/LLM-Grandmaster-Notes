# KV Cache Sparse

## 介绍

KV Cache Sparse（稀疏 KV Cache）是减少 KV Cache 显存占用的一类方法，通过只保留"重要"的 KV 对，丢弃或压缩不重要 token 的 KV 缓存。

## 核心思想

在自回归推理中，并非所有历史 token 对当前生成同等重要。稀疏 KV Cache 方法选择性地保留关键 token 的 KV，而非保留全部。

## 主要方法

### 1. H2O (Heavy-Hitter Oracle)

[H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)

核心观察：少量"核心 token"（Heavy-Hitter）在注意力中贡献了大部分权重。

策略：

- 保留最近 $k$ 个 token 的 KV（局部性保证）
- 保留累积注意力权重最高的 $h$ 个 token 的 KV（全局重要性）
- 缓存预算 = $k + h$

$$
\text{KV\_Cache} = \text{Recent}(k) \cup \text{HeavyHitter}(h)
$$

驱逐策略：当缓存满时，驱逐累积注意力分数最低的非最近 token。

### 2. StreamingLLM

[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)

核心观察：Attention Sink 现象——大量注意力权重集中在序列开头的几个 token 上（即使它们语义上不重要）。

策略：

- 固定保留序列开头的 $s$ 个 token（Attention Sink）
- 保留最近 $w$ 个 token（滑动窗口）
- 缓存预算 = $s + w$

$$
\text{KV\_Cache} = \text{Sink}(s) \cup \text{Window}(w)
$$

### 3. Scissorhands

[Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time](https://arxiv.org/abs/2305.17118)

策略：通过分析注意力模式，识别对当前生成影响最大的 token，保留这些 token 的 KV 缓存。

### 4. Key-Aware Eviction

基于 Key 本身的重要性分数决定驱逐哪些 token：

- 计算每个 Key 的"重要性"（如注意力分数的移动平均）
- 当缓存满时，驱逐重要性最低的 Key

## 稀疏 KV Cache 的权衡

| 指标 | 全量 KV Cache | 稀疏 KV Cache |
|:---:|:---:|:---:|
| 显存占用 | $O(N)$ per layer | $O(B)$ per layer（$B$ 为预算）|
| 模型质量 | 最优 | 有损（取决于稀疏策略）|
| 适用序列长度 | 受显存限制 | 更长 |
| 推理速度 | 受访存限制 | 更快（更少的 KV 读取）|

## 实际应用

稀疏 KV Cache 在超长上下文推理场景中尤为重要：

- 当序列长度超过 100K+ 时，全量 KV Cache 显存不可承受
- H2O 和 StreamingLLM 等方法可以在仅保留 5-10% KV 的情况下保持大部分性能
