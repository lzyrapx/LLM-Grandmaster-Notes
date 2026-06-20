# Grouped-Query Attention (GQA)

## paper

[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)

## 介绍

Grouped-Query Attention (GQA) 是 MHA 和 MQA 之间的折中方案。MHA 每个 Query 头都有独立的 Key/Value 头，MQA 所有 Query 头共享一个 Key/Value 头，而 GQA 将 Query 头分成若干组，每组共享一组 Key/Value 头。

## MHA vs GQA vs MQA 对比

| 类型 | Q 头数 | K 头数 | V 头数 | KV Cache | 代表模型 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| MHA | $h$ | $h$ | $h$ | 最大 | GPT-3, LLaMA-1 |
| GQA | $h$ | $g$ ($1 < g < h$) | $g$ | 中等 | LLaMA-2/3, Gemma |
| MQA | $h$ | $1$ | $1$ | 最小 | PaLM, Falcon |

其中 $h$ 是总 Query 头数， $g$ 是 KV 组数。

## 数学公式

假设将 $h$ 个 Query 头分成 $g$ 组，每组有 $h/g$ 个 Query 头共享同一组 KV 头：

$$
\text{head}_i = \text{Softmax}\left(\frac{Q_i K_{\lfloor i \cdot g / h \rfloor}^T}{\sqrt{d_k}}\right) V_{\lfloor i \cdot g / h \rfloor}
$$

其中 $i$ 是 Query 头的索引， $\lfloor i \cdot g / h \rfloor$ 将 Query 头映射到对应的 KV 组。

## PyTorch 实现

```python
import torch
import torch.nn as nn
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # 每个 KV 头对应的 Q 头数

        self.W_Q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        Q = self.W_Q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)     # (B, n_heads, S, d_k)
        K = self.W_K(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, n_kv_heads, S, d_k)
        V = self.W_V(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, n_kv_heads, S, d_k)

        # 扩展 KV 头以匹配 Q 头数: (B, n_kv_heads, S, d_k) -> (B, n_heads, S, d_k)
        K = K.repeat_interleave(self.n_rep, dim=1)
        V = V.repeat_interleave(self.n_rep, dim=1)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, n_heads, S, d_k)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_O(out)
```

## 从 MHA 上转换到 GQA

GQA 论文的一个重要贡献是提出了从已训练好的 MHA 模型转换为 GQA 模型的方法：

1. **mean conversion**：将同一组内的 KV 头取平均
2. **choose-one conversion**：选择每组中的头作为代表

转换后进行少量 fine-tuning 即可恢复大部分性能，这避免了从零训练 GQA 模型的巨大成本。

## KV Cache 节省

GQA 相比 MHA，KV Cache 减少比例为：

$$
\text{节省比例} = \frac{MHA KV头数 - GQA KV头数}{MHA KV头数} = \frac{h - g}{h} = 1 - \frac{g}{h}
$$

**$h$** 是 Query（查询）的注意力头数。在标准的 MHA 中，Key/Value 的注意力头数与 Query 一致（即 $h_{kv} = h$）。

**$g$** 是 GQA 或 MQA（Multi-Query Attention）中实际的 Key/Value（KV）注意力头数（其中 $g \le h$）

例如：

- Gemma 3 12B：$h=16, g=8$，KV Cache 减少了 $50$%。
- Gemma 4 12B：
    - sliding 层：$h=16, g=8$，KV Cache 减少了 $50$%。
    - global 层：$h=16, g=1$，KV Cache 减少了 $93.75$%。

注意：在实际工程部署中，Gemma 3 和 Gemma 4 的 KV Cache 节省比例远比单纯由上面的公式计算出的比例还要高，原因在于它们不仅在“注意力头”维度上进行分组（GQA/MQA），还在"序列长度 $N$" 维度上进行了精简，比如 Sliding Windows Attention 机制等。
