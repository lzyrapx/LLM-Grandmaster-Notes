# KV Cache Window

## 介绍

KV Cache Window（滑动窗口 KV Cache）是一种限制 KV Cache 大小的策略，只保留最近 $W$ 个 token 的 Key 和 Value，丢弃更早的 KV 对。

## 核心思想

在自回归推理中，不是所有历史 token 都同等重要。近期 token 的 KV 对当前生成影响通常更大。滑动窗口策略只保留最近 $W$ 个 token：


在自回归生成的位置 $t$，其 query $q_t$ 在计算注意力时，只关注其自身及最近的 $W - 1$ 个历史 token：

$$
O_t = \text{softmax}\left(\frac{q_t [k_{t-W+1}, \ldots, k_t]^T}{\sqrt{d_k}}\right) [v_{t-W+1}, \ldots, v_t]
$$

此时实际的注意力窗口大小为 $\min(t, W)$

## 显存占用

滑动窗口将 KV Cache 的显存占用从 $O(N)$ 降到 $O(W)$：

$$
\text{KV Cache Size} = 2 \times B \times L \times W \times n_{kv} \times d_h \times \text{sizeof(dtype)}
$$

其中 $B$ 为 Batch Size， $L$ 为层数， $W$ 为窗口大小， $n_{kv}$ 为 KV Head 数量， $d_h$ 为每个 Head 的维度）。

当 $W \ll N$ 时，显存节省非常显著。

## 代表模型

### Mistral / Gemma (Sliding Window Attention)

Mistral-7B v0.1 和部分 Gemma 模型使用了滑动窗口注意力：

- **Mistral-7B v0.1** : $W = 4096$，即每个 attention layer 只关注最近 4096 个 token
    - 只有 **Mistral-7B v0.1** 在预训练和推理中使用了标准的滑动窗口注意力（SWA, $W=4096$）
    - 在后来的 **Mistral-7B v0.2**（以及后来的很多 Mistral 变体）中，官方**移除了滑动窗口注意力**，直接将上下文硬支持扩展到了 32k。因为纯滑动窗口会导致模型长文本检索能力（如 Needle In A Haystack）大幅下降。
- 训练时使用滑动窗口，推理时 KV Cache 大小恒定

- 关于 Gemma：
    - Gemma 不是纯滑动窗口，而是典型的**混合/交替窗口策略（Hybrid/Interleaved Attention）**
    - **Gemma 2**：在局部滑动窗口层（ $W=4096$ ）与全局注意力层之间进行 **1:1** 交替（一层滑动，一层全局）
    - **Gemma 3 / 4**：为了处理高达 128K/256K 的长上下文并极大地压缩 KV Cache，它们采用了更激进的 **5:1** 交替策略（5层 $W=1024$ 的局部滑动窗口，交替 1 层全局层），将 KV Cache 显存开销压缩至原本的几分之一。

### 长序列的局限

纯滑动窗口的问题：

- 超出窗口 $W$ 的信息完全丢失
- 无法捕捉长距离依赖
- 对需要全文信息的任务（如文档 QA）有损

## 改进：混合窗口策略

### 1. 分层窗口（Layer-wise Window）

不同层使用不同的窗口大小：

- 浅层使用小窗口（捕捉局部模式）
- 深层使用大窗口（捕捉全局信息）

### 2. 窗口 + 全局 Token

保留少量全局 token（如序列开头的 sink token）+ 滑动窗口：

$$
\text{KVCache} = \text{Global}(G) \cup \text{Window}(W)
$$

类似 StreamingLLM 的思路。

### 3. 窗口 + 稀疏选择

在窗口之外，使用稀疏选择策略保留少数重要 token：

$$
\text{KVCache} = \text{Important}(K) \cup \text{Window}(W)
$$

类似 H2O 的思路。

## 与 PagedAttention 的配合

滑动窗口 + PagedAttention 的组合：

- 窗口滑动时，最老的 block 整体释放
- 新 block 按需从 block pool 分配
- 保持活跃 block 数恒定

## 参考资料

- https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4
- Gemma2: https://arxiv.org/pdf/2408.00118v1
- Gemma3: https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/
- Gemma4: https://ai.google.dev/gemma/docs/core/model_card_4?hl=zh-cn
- https://devopslearning.medium.com/building-gemma-3-from-scratch-323112c544e2
- https://colab.research.google.com/drive/1e61rS-B2gsYs_Z9VmBXkorvLU-HJFEFS?usp=sharing