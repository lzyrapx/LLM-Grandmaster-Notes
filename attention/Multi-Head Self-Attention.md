## 总结

- Multi-Head Self-Attention (`MHSA`) 是 `MHA` 的一个子集，特指 `Q`, `K`, `V` 都来自**同一个输入序列 `X` 的情况**。它的核心作用是计算序列内部元素之间的关系。
- 当 `Q`, `K`, `V` 来源于不同序列时，就是更通用的 `MHA`，其核心作用是计算不同序列之间元素的关系。
- `Multi-Head Self-Attention (MHSA)` 是 `Multi-Head Attention (MHA)` 的一个特例。
- 很多开源代码里，类名叫 `MultiHeadAttention`（通用术语），但具体实现是 `Multi-Head Self-Attention`，基本上 `query`、`key`、`value` 都直接通过对同一个 `hidden_state` 做线性变换得到。如下：

    ```
    query = self.q_linear(hidden_state)
    key = self.k_linear(hidden_state)
    value = self.v_linear(hidden_state)
    ```

## 基础

### Scaled Dot-Product Attention (SDPA, 单头注意力)

在理解多头之前，先回顾一下单头注意力 `SDPA` 的计算 `Attention(Q, K, V)`:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^{T}}{\sqrt{d_k}})V
$$

- 输入表示： 假设有一个输入序列，包含 `n` 个元素（例如单词），每个元素被表示为一个 `d_model` 维的向量。我们将整个序列堆叠成一个矩阵 `X` (形状 `n x d_model`)
- 线性投影 (`Query`, `Key`, `Value`)：
  - 使用三个不同的可学习权重矩阵 `W^Q`, `W^K`, `W^V` (每个形状 `d_model x d_k` 或 `d_model x d_v`，通常 `d_k = d_v`)。
  - `Query` 向量： `Q = X * W^Q` (形状 `n x d_k`) - 代表我们要“询问”序列中哪些部分信息。
  - `Key` 向量： `K = X * W^K` (形状 `n x d_k`) - 代表序列中每个元素可以用来被“查询”的标识。
  - `Value` 向量： `V = X * W^V` (形状 `n x d_v`) - 代表序列中每个元素实际包含的信息。
- 计算注意力分数（`Dot-Product`）： 计算 `Query` 向量和所有 `Key` 向量之间的相似度（点积）。分数矩阵 `S = Q * K^T` (形状 `n x n`)。
- 缩放（`Scaling`）：将注意力分数除以 `sqrt(d_k)`。这一步非常重要，因为点积在 `d_k` 较大时其值会非常大，导致 `softmax` 梯度极小，防止梯度爆炸或消失。
- `Softmax` 归一化： 对缩放后的分数矩阵的每一行应用 `softmax` 函数。这使得每一行的分数总和为 `1`，并产生 注意力权重矩阵 `A` (形状 `n x n`)。`A[i, j]` 表示第 `i` 个元素在生成其输出表示时，应该给予第 `j` 个元素多少“注意力”。
- 加权求和： 将 `Value` 矩阵 `V` 用注意力权重 `A` 加权求和，得到输出矩阵 `Z` (形状 `n x d_v`)： `Z = A * V`。
  - `Z[i] = sum_j(A[i, j] * V[j])` - 第 `i` 个元素的输出是其自身 `Value` 和所有其他元素 `Value` 的加权和，权重由它与所有其他元素的相关性决定。

`SDPA` 的意义： 它允许序列中的每个位置在生成自己的新表示时，能够“关注”序列中所有其他位置（包括自身）的信息，并根据相关性动态地聚合这些信息。

## Multi-Head Self-Attention (MHSA, 多头自注意力)

`SDPA` 只有一个“视角”去关注序列。`MHSA` 的核心思想是：并行地执行多次（`h` 次）不同的注意力计算，每次关注不同方面的信息，然后将所有结果组合起来。

1. 投影到多个子空间：
    - 不再直接将 `X` 投影到 `d_k` 维的 `Q`, `K`, `V`。 
    - 而是将 `X` 分别投影 `h` 次（每个头一次），每次使用独立的可学习权重矩阵 `W_i^Q`, `W_i^K`, `W_i^V` (每个形状 `d_model x (d_model / h)`)。这里 `d_model` 必须能被 `h` 整除。
    - 得到 `h` 组 `Query`, `Key`, `Value` 矩阵：
        - `Q_i = X * W_i^Q` (形状 `n x (d_model / h)`)
        - `K_i = X * W_i^K` (形状 `n x (d_model / h)`)
        - `V_i = X * W_i^V` (形状 `n x (d_model / h)`)
    - 并行计算 `h` 个注意力头：
        - 对每一组 `Q_i`, `K_i`, `V_i`，独立地执行前面描述的 `Scaled Dot-Product Attention` 操作 (`Attention(Q_i, K_i, V_i)`)。
        - 每个头产生一个输出矩阵 `Z_i` (形状 `n x (d_model / h)`)。
    - 拼接 (`Concat)`：
        - 将所有 `h` 个头的输出 `Z_i` 按列拼接 (`concat`) 起来，形成一个大的矩阵 `Z_concat` (形状 `n x d_model`)。
    - 线性投影：
        - 使用一个可学习的权重矩阵 `W^O` (形状 `d_model x d_model`) 对拼接后的 `Z_concat` 进行线性变换。
        - 得到最终的输出矩阵 `Z` (形状 `n x d_model`)： `Z = Z_concat * W^O`。

`“Self”` 在 `MHSA` 中的含义：

- 键点在于：`Query`, `Key`, `Value` 三个向量都来源于同一个输入序列 `X`。
- 它计算的是一个序列内部元素之间的相互依赖关系（自省）。例如，在编码器中，MHSA 让序列中的每个单词都能关注序列中的所有其他单词，以建立更好的上下文表示。

## Multi-Head Self-Attention (MHSA) 与 Multi-Head Attention (MHA) 的区别

- Multi-Head Self-Attention (MHSA) 是 Multi-Head Attention (MHA) 的一个特例。
- 核心区别在于 `Query`, `Key`, `Value` 的来源：

|特性|Multi-Head Self-Attention|Multi-Head Attention|
|:---:|:---:|:---:|
|`Q` 来源|输入序列 `X` (自身)|一个序列 `X` (通常是目标序列)|
|`K` 来源|输入序列 `X` (自身)|另一个序列 `Y` (通常是源序列)|
|`V` 来源|输入序列 `X` (自身)|另一个序列 `Y` (通常是源序列)|
|目的|计算同一个序列内部元素之间的关系|计算两个不同序列之间元素的关系|
|典型应用|Transformer 解码器层的第一个注意力层|Transformer 解码器层的第二个注意力层|
|关系|`MHA` 的一种特殊情况 (当 `X` = `Y`)|通用形式|

## Pytorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        # Dropout 是一种正则化技术，用于防止神经网络过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: [batch_size, num_heads, seq_len, d_k]
        d_k = Q.size(-1)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        # 应用掩码（如需要）
        # 添加 Mask，mask 中为 0 的位置不受影响，为 -1e9 的位置被屏蔽
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        # x: [batch_size, num_heads, seq_len, d_k]
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    # MHSA 只有一个输入 x
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]
        
        # 线性投影
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 分割多头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 计算注意力
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # 合并多头
        attn_output = self.combine_heads(attn_output)
        
        # 输出投影
        output = self.W_o(attn_output)
        return output, attn_weights
```