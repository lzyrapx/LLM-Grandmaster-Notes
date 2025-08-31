## Multi-Head Attention (MHA)

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^{T}}{\sqrt{d_k}})V
$$

$$
MultiHead(Q, K, V)=Concat(head_{1},...,head_{h})W^{O}
$$

其中，

$$
head_{i}=Attention(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V})
$$

- 这是一个更通用的概念，`MHSA` 是 `MHA` 的子集。
- `Q` 可以来自一个序列 `X`，而 `K` 和 `V` 来自另一个序列 `Y`。
    - `Q = X * W^Q`
    - `K = Y * W^K`
    - `V = Y * W^V`
- 目的： 让序列 `X` 中的每个元素（通常是目标序列）去 “查询” 序列 `Y`（通常是源序列）中的相关信息。这是一种跨序列的注意力机制。
    - `Q` 来自解码器上一层的输出（目标序列的表示）。
    - `K` 和 `V` 来自编码器最终的输出（源序列的表示）。
- 应用： 在 `Transformer` 的解码器中，第二个注意力层（`Encoder-Decoder Attention`）就是典型的 `MHA`（不是 `MHSA`）：
- **注意**：`Transformer`里，大部分 `forward` 的具体实现是 `MHSA`，但也可以更通用地叫 `MHA`。
    - 严格意义上， `Transformer`里 只有 `Cross-Attention` 是 `MHA`，比如：[modeling_mega](https://github.com/huggingface/transformers/blob/2c0af41ce5c448f872f3222a75f56030fb2e5a88/src/transformers/models/deprecated/mega/modeling_mega.py#L1270)

- 总的来说，`Multi-head Attention` 就是把序列中的每个 `token` 的表示经过线性映射到不同的子空间中，然后在不同的子空间中计算 `attention`，最后把结果拼接在一起并经过一个线性层来综合不同子空间的上下文表示，得到最终的 `Multi-head Attention` 的输出。

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

class MultiHeadAttention(nn.Module):
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

    def forward(self, Q, K, V, mask=None):
        # Q: [batch_size, q_seq_len, d_model]
        # K, V: [batch_size, kv_seq_len, d_model]
        
        # 线性投影
        Q_proj = self.W_q(Q)
        K_proj = self.W_k(K)
        V_proj = self.W_v(V)
        
        # 分割多头
        Q_heads = self.split_heads(Q_proj)
        K_heads = self.split_heads(K_proj)
        V_heads = self.split_heads(V_proj)
        
        # 计算注意力
        attn_output, attn_weights = self.attention(Q_heads, K_heads, V_heads, mask)
        
        # 合并多头
        attn_output = self.combine_heads(attn_output)
        
        # 输出投影
        output = self.W_o(attn_output)
        return output, attn_weights
```