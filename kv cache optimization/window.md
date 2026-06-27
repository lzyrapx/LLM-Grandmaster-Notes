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

## 其他

#### 堆叠滑动窗口的“有效感受野”与“信息稀释”

- 多层传递效应
    在 Transformer 中，单层滑动窗口的视野只有 $W$。但因为网络有多层（ $L$ 层），在第 $l$ 层时，Token 实际上可以通过前一层的隐状态间接获取更早的信息。理论上，网络的 **最大有效感受野（Effective Receptive Field）** 可以线性累加到 $L \times W$。
- **现实局限（信息稀释）**
    尽管理论上感受野随着层数加深而扩大，但在实际应用中，由于 **残差连接（Residual Connections）和注意力权重的稀释（Information Dilution）** ，信息在多层 SWA 之间传递时会像“传话游戏”一样迅速衰减。研究表明，纯 SWA 模型在几千个 Token 之外的精细召回率（如 Passkey 任务）会呈指数级下降，这也是为什么现代模型（如 Gemma 3/4）必须引入全局注意力层（Global Layers）来进行信息锚定的原因。

#### 为什么滑动窗口必须搭配 Attention Sink？（数学原理）

上面提到了 StreamingLLM（窗口 + 全局 Token），背后的 **数学和物理机制** 是：

* **Softmax 溢出与垃圾回收机制** ：
  Softmax 函数要求所有分量的指数和为 1： $\sum e^{s_i} = 1$。

  在自回归模型中，当生成的序列很长时，很多新生成的 Token 与历史 Token 之间并没有强语义关联，但 Softmax 强制要求分配注意力权重。此时，模型在训练中学会了将 **第一或前几个 Token（通常是 `<s>` 或首个单词）** 作为“垃圾回收桶”（Attention Sink），倾倒不必要的注意力权重。
* **如果不保留 Sink 会发生什么** ：
  如果采用纯滑动窗口丢弃了最初的几个 Token，后期的 Query 找不到可以“倾倒”垃圾注意力的对象，会导致 Softmax 分母异常，注意力分数异常暴涨，进而导致模型的困惑度（Perplexity）发生灾难性飙升。保留 1~4 个起始 Token 即可完全避免这一现象。
  
  简单来说就是：在滑动窗口过程中，一旦丢弃了开头 Token 后，新生成的词由于 Softmax 的限制，必须强行分配注意力，却失去了那个在训练中习惯用来容纳无意义注意力的首个 Token。这迫使它将注意力乱投给当前窗口的其他词，进而污染了正常的语义特征，导致模型崩溃。

   StreamingLLM 的解决方案非常简单且优雅：永远在窗口中保留最开头的 1~4 个 Token 不丢弃。只要保留了这个“垃圾桶”，滑动窗口就可以无限滑动下去，而模型绝不会崩溃。
  
## 参考资料

- https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4
- Gemma2: https://arxiv.org/pdf/2408.00118v1
- Gemma3: https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/
- Gemma4: https://ai.google.dev/gemma/docs/core/model_card_4?hl=zh-cn
- https://devopslearning.medium.com/building-gemma-3-from-scratch-323112c544e2
- https://colab.research.google.com/drive/1e61rS-B2gsYs_Z9VmBXkorvLU-HJFEFS?usp=sharing
- https://guangxuanx.com/blog/stacking-swa.html