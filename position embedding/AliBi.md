# AliBi (Attention with Linear Biases)

## paper

[Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)

## 介绍

AliBi 是一种无需位置编码的方法，通过在注意力分数上添加线性偏置来实现位置感知。AliBi 的核心优势是**不需要训练时看到长序列也能在推理时外推到更长序列**。

## 核心思想

AliBi 不在输入 embedding 上添加位置编码，也不对 Q/K 做旋转变换，而是直接在注意力分数矩阵上添加一个与相对位置成比例的线性偏置：

$$
 \text{softmax}(q_i k_j^T - m \cdot |i - j|) 
$$

对于 causal attention（ $i < j$ 时不允许注意），偏置为：

$$
\text{softmax}(q_i k_j^T - m \cdot (i - j)), \quad i \geq j
$$

其中 $m$ 是每个注意力头独有的斜率（slope），是一个不可学习的超参数。

注意偏置是**负数**，距离越远的 token 偏置越大（惩罚越大），使得模型更关注近处 token。

## 斜率 $m$ 的设定

AliBi 为每个注意力头设定不同的斜率，形成几何级数：

$$
m_h = \frac{1}{2^{\frac{8}{H} \cdot h}}, \quad h = 1, 2, \ldots, H
$$

其中 $H$ 是总头数。

例如 $H = 8$ 时：

$$
m = \{\frac{1}{2^1}, \frac{1}{2^2}, \frac{1}{2^3}, \frac{1}{2^4}, \frac{1}{2^5}, \frac{1}{2^6}, \frac{1}{2^7}, \frac{1}{2^8}\} = \{\frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}, \frac{1}{64}, \frac{1}{128}, \frac{1}{256}\}
$$

不同头关注不同距离范围：斜率大的头关注近处，斜率小的头关注远处。

## 与 RoPE 的对比

| 特性 | RoPE | AliBi |
|:---:|:---:|:---:|
| 位置信息注入 | 旋转 Q, K | 注意力分数偏置 |
| 可学习 | 否（固定公式）| 否（固定斜率）|
| 长度外推 | 差（需额外方法）| **好（原生支持）** |
| 远程衰减 | 有（高频衰减快）| 有（线性偏置）|
| 计算开销 | 逐元素旋转 | 添加偏置矩阵 |
| 代表模型 | LLaMA, Gemma, Mistral | BLOOM, MPT |

## AliBi 的注意力偏置矩阵

对于序列长度 $N$，AliBi 的偏置矩阵为：

$$
B_{ij} = -m_h \cdot |i - j|
$$

```
示例 (m_h = 0.5):
     j=0  j=1  j=2  j=3
i=0 [ 0   -0.5 -1.0 -1.5]
i=1 [-0.5  0   -0.5 -1.0]
i=2 [-1.0 -0.5  0   -0.5]
i=3 [-1.5 -1.0 -0.5  0  ]
```

对于 causal mask（只看当前位置及之前）：

```
Causal AliBi (m_h = 0.5):
     j=0  j=1  j=2  j=3
i=0 [ 0   -inf -inf -inf]
i=1 [-0.5  0   -inf -inf]
i=2 [-1.0 -0.5  0   -inf]
i=3 [-1.5 -1.0 -0.5  0  ]
```

## 长度外推

AliBi 的线性偏置天然支持长度外推：

- 训练时使用序列长度 $L_{\text{train}}$
- 推理时可以直接处理 $L_{\text{test}} > L_{\text{train}}$ 的序列
- 线性偏置对更远的距离给予更大的惩罚，防止注意力分配到远距离噪声 token

实验中，AliBi 在训练序列长度 1024 的情况下，可以外推到 2048+ 的序列，且困惑度增长缓慢。

## PyTorch 实现

```python
import math
import torch

def get_alibi_slopes(num_heads):
    """
    计算每个头的斜率 (slope)
    """
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        closest_power = 2 ** math.floor(math.log2(num_heads))
        return (get_slopes_power_of_2(closest_power) +
                get_slopes_power_of_2(2 * closest_power)[0::2][:num_heads - closest_power])

def get_alibi_bias(num_heads: int, seq_len: int, device="cpu") -> torch.Tensor:
    """
    生成适用于非因果（Symmetric）或配合 Causal Mask 共同使用的 AliBi 偏置矩阵
    返回维度: (1, num_heads, seq_len, seq_len)
    """
    slopes = get_alibi_slopes(num_heads)
    slopes_tensor = torch.tensor(slopes, device=device).view(1, num_heads, 1, 1)

    # 相对位置距离矩阵
    # shape: (seq_len, seq_len)
    range_vec = torch.arange(seq_len, device=device)
    distances = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)  # j - i
    distances = distances.abs().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    # 乘以负斜率得到偏置矩阵
    alibi_bias = -slopes_tensor * distances.float()
    return alibi_bias
```

## 其他

### 1. 为什么 AliBi 能够实现“长度外推”？（数学本质）

- **Softmax 的平移不变性**：
  Softmax 函数满足 $\text{softmax}(x) = \text{softmax}(x + c)$。
  在因果语言模型（Causal LM）中，第 $i$ 个 token 的自注意力计算形如 $\text{softmax}(q_i k_j^T - m \cdot (i - j))$（其中 $j \le i$）。
  当我们从训练长度 $L_{\text{train}}$ 外推到更长的推理长度 $L_{\text{test}}$ 时，新增的远距离 token 的偏置项 $- m \cdot (i - j)$ 会非常大（如 $-32$、 $-64$ 等），这会导致这些远距离 token 的指数项 $e^{\text{score}}$ 趋于 $0$。
  得益于此，新加入的超长距离 token 不会破坏已有近距离 token 的注意力分配权重，模型因此能够保持相对稳定的困惑度（Perplexity）。相比之下，旋转位置编码（RoPE）在面对未曾训练过的超长距离时，其 Query-Key 的旋转向量可能产生不稳定的高频振荡，导致外推失败。

### 2. 因果场景（Causal Attention）下的无绝对值（`.abs()`）优化

在实际的大模型因果自注意力代码中，我们常常不需要执行 `.abs()` 操作，而是直接使用：

```python
# 相对距离为：j - i (其中 j 是 key 指针，i 是 query 指针)
# 在 Causal 下，当 j > i 时，该值 > 0，乘以 -m 会在未来部分产生负偏置（如 -0.5, -1.0）。
# 但由于未来部分会被 Causal Mask 覆盖（加上 -inf），
# 因此未来位置的值是否准确并不影响最终的 Softmax 结果。
distances = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
```

官方库利用了这一性质，省略了 `.abs()` 计算，节省了操作指令，同时也让矩阵运算在硬件上更加整洁。

### 3. AliBi 与 FlashAttention 的兼容性与硬件开销

- **显存开销问题**：
  在原始 Transformer 中，注意力掩码（Mask）是各头共享的，尺寸为 $1 \times L \times L$。
  而 AliBi 需要为每个注意力头乘上不同的斜率 $m$，因此其偏置矩阵尺寸为 $H \times L \times L$。当序列极长、头数较多时，显存占用会有所增加。

- **与 FlashAttention 的集成**：
  早期的 FlashAttention 不支持动态偏置，这导致使用 AliBi 的模型（如 BLOOM、MPT）很难享受早期的硬件加速。
  不过，后来的硬件加速库（如 FlashAttention-2、Triton、vLLM）支持在 GPU 的 SRAM 中**在线计算（On-the-fly）** AliBi 的线性偏置，即直接将 `alibi_slopes`（长度为 $H$ 的一维向量）传入算子中，在计算注意力分数的同时根据线程索引动态减去 $m \cdot |i-j|$，从而既实现了极速推理，又避免了在显存中存储 $H \times L \times L$ 的偏置矩阵。

### 4. 为什么近年来的主流模型（LLaMA / Gemma / Mistral）更倾向于使用 RoPE？

尽管 AliBi 在外推性能上表现极佳，但目前绝大多数新模型（如 LLaMA、Gemma 等）仍采用了 RoPE，主要原因如下：

- **表达能力的局限性**：AliBi 强制对距离施加了严格的、不可学习的**线性衰减**。这限制了模型捕获复杂、非单调的位置关系的能力（例如，有些注意力头可能需要重点关注开头或某些特定固定跨度的 Token，而不需要严格的距离衰减）。

- **RoPE 外推方案的成熟**：在 AliBi 提出后，又为 RoPE 研发出了 YaRN、NTK-aware Scaling、RoPE-BASE 动态调整等优秀的插值与外推方案，成功解决了 RoPE 的外推痛点，使得 RoPE 在保留高表达能力的同时也具备了极佳的上下文扩展能力。

## 参考资料

- https://www.abhik.ai/concepts/transformers/alibi
- https://nn.labml.ai/transformers/alibi/index.html
- https://disassemble-channel.com/alibi-positional-encoding/
- https://towardsdatascience.com/positional-embeddings-in-transformers-a-math-guide-to-rope-alibi/
- https://proceedings.neurips.cc/paper_files/paper/2022/file/37a413841a614b5414b333585e7613b8-Supplemental-Conference.pdf
- https://iclr-blogposts.github.io/2025/blog/positional-embedding/