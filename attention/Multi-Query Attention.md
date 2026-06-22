# Multi-Query Attention (MQA)

## paper

[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)

## 介绍

Multi-Query Attention (MQA) 是 MHA 的极端简化版本：所有 Query 头共享**同一组** Key 和 Value 头。这样在推理时，KV Cache 只需存储一套 K 和 V，极大降低了显存占用和访存开销。

## MHA vs MQA

| 类型 | Q 头数 | K 头数 | V 头数 | KV Cache 大小 |
|:---:|:---:|:---:|:---:|:---:|
| MHA | $h$ | $h$ | $h$ | $2 \times L \times s \times h \times d_h$ |
| MQA | $h$ | $1$ | $1$ | $2 \times L \times s \times 1 \times d_h$ |

MQA 相比 MHA，KV Cache 减少了 $\frac{h-1}{h}=1-\frac{1}{h}$。

* $L$：Transformer 的层数（Layers）
* $s$：序列长度（Sequence Length）
* $h$：Query 头数（Heads）
* $d_h$：每个头的维度（Head Dimension）
* $b$：每个参数的字节大小（Byte size，如 FP16/BF16 为 2 字节，FP32 为 4 字节，INT8 为 1 字节）。
* **以 FP16 为例，MQA 的物理显存占用公式为**：
  $$\text{KV Cache Size (Bytes)} = 4 \times L \times s \times 1 \times d_h \quad (\text{因为 } 2 \text{个矩阵 } [K, V] \times 2 \text{ 字节} = 4)$$
## 数学公式

$$
\text{head}_i = \text{Softmax}\left(\frac{Q_i K^T}{\sqrt{d_k}}\right) V
$$

所有 $h$ 个 Query 头使用相同的 $K$ 和 $V$（形状为 $(1, S, d_k)$ 和 $(1, S, d_v)$）。

## 训练或 prefill 的 PyTorch 实现

```python
import torch
import torch.nn as nn
import math

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_Q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(d_model, self.head_dim, bias=False)       # 只有一个 KV 头
        self.W_V = nn.Linear(d_model, self.head_dim, bias=False)       # 只有一个 KV 头
        self.W_O = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        Q = self.W_Q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, S, d_k)
        K = self.W_K(x).view(B, S, 1, self.head_dim).transpose(1, 2)             # (B, 1, S, d_k)
        V = self.W_V(x).view(B, S, 1, self.head_dim).transpose(1, 2)             # (B, 1, S, d_k)

        # 虽然 expand 不会真正复制物理内存（只改变 stride）
        # 但实际上由于 PyTorch 的 torch.matmul 具备自动广播机制（Broadcasting），这两行代码是可以省略的。
        # 当 Q 的形状为 (B, n_heads, S, d_k)，K.transpose(-2, -1) 的形状为 (B, 1, d_k, S) 时，
        # torch.matmul(Q, K.transpose(-2, -1)) 会自动将 K 在头数维度上从 1 广播到 n_heads。
        
        # 扩展 K, V 以和所有 Q 头做注意力
        K = K.expand(B, self.n_heads, S, self.head_dim)   # (B, n_heads, S, d_k)
        V = V.expand(B, self.n_heads, S, self.head_dim)   # (B, n_heads, S, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_O(out)
```

## 带有 KV Cache 状态的推理（Decode）的 Pytorh 实现

```python
import torch
import torch.nn as nn
import math

class MQA_Decoding(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_Q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(d_model, self.head_dim, bias=False)       # 只有一个 KV 头
        self.W_V = nn.Linear(d_model, self.head_dim, bias=False)       # 只有一个 KV 头
        self.W_O = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_key=None, past_value=None):
        # 推理 Decode 时，x 的形状通常为 (B, 1, d_model)
        B, S, _ = x.shape
        
        Q = self.W_Q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, 1, d_k)
        K = self.W_K(x).view(B, S, 1, self.head_dim).transpose(1, 2)             # (B, 1, 1, d_k)
        V = self.W_V(x).view(B, S, 1, self.head_dim).transpose(1, 2)             # (B, 1, 1, d_k)

        # 拼接历史 KV Cache
        if past_key is not None:
            K = torch.cat([past_key, K], dim=2)   # 沿序列长度维度(dim=2)拼接 -> (B, 1, S_seq, d_k)
            V = torch.cat([past_value, V], dim=2) # -> (B, 1, S_seq, d_k)
        
        # 保存新的 KV Cache 供下一步使用
        present_key, present_value = K, V

        # 计算 Attention（利用 PyTorch 自动广播 K 和 V 维度）
        # Q: (B, n_heads, 1, d_k) 乘以 K_T: (B, 1, d_k, S_seq) -> scores: (B, n_heads, 1, S_seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        
        # attn: (B, n_heads, 1, S_seq) 乘以 V: (B, 1, S_seq, d_k) -> out: (B, n_heads, 1, d_k)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        
        return self.W_O(out), (present_key, present_value)
```

## 推理加速原理

在 Decode 阶段，每个 token 生成都需要从 HBM 读取 KV Cache。MQA 的优势：

1. **KV Cache 体积缩小**：从 $h$ 组 K/V 减少为 1 组
2. **访存量大幅减少**：Decode 阶段是 memory-bound，KV Cache 越小，读取延迟越低
3. **更高的 batch size**：节省的显存可用于增大 batch size，提高吞吐

## 优缺点

**优点**：
- KV Cache 极小，推理速度快
- Decode 阶段访存量接近最优

**缺点**：
- 相比 MHA，模型质量可能下降（尤其是在复杂推理任务上）
- 所有 Q 头共享完全相同的 KV，表达力下降
- GQA 是 MQA 和 MHA 的更好折中


## MHA $\to$ GQA $\to$ MQA 的演进

为了将 GQA 和 MQA 联系起来，给出一个直观的设计对比：

$$\text{MHA (每个 Q 配一个独立的 K, V)} \quad \longleftrightarrow \quad \text{GQA (多个 Q 共享一组 K, V)} \quad \longleftrightarrow \quad \text{MQA (所有 Q 共享一组 K, V)}$$

* **多头注意力 (MHA)**： $h$ 个 Q 头， $h$ 个 KV 头。
* **分组查询注意力 (GQA)**： $h$ 个 Q 头，被均分为 $g$ 组（Groups），每组内的 Query 头共享一个 KV 头，共有 $g$ 个 KV 头。
  * 当 $g=h$ 时，退化为 MHA。
  * 当 $g=1$ 时，退化为 MQA。