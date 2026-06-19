# Transformer

## paper

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 介绍

Transformer 是一种基于自注意力机制（Self-Attention）的序列到序列模型架构，由 Google 在 2017 年提出，完全抛弃了 RNN 和 CNN 的结构，仅靠注意力机制来建模序列间的依赖关系。

## 整体架构

Transformer 由 **Encoder** 和 **Decoder** 两部分组成：

- **Encoder**：由 N 个相同的 Encoder Layer 堆叠而成，每个 Layer 包含 Multi-Head Self-Attention + Feed-Forward Network
- **Decoder**：由 N 个相同的 Decoder Layer 堆叠而成，每个 Layer 包含 Masked Multi-Head Self-Attention + Cross-Attention + Feed-Forward Network

```
Input → [Embedding + Pos Encoding] → Encoder × N → [K, V]
                                                    ↓
Output → [Embedding + Pos Encoding] → Decoder × N → Linear → Softmax → Output Prob
```

## Encoder

每个 Encoder Layer 包含两个子层：

1. **Multi-Head Self-Attention**：序列内各位置之间的注意力
2. **Feed-Forward Network (FFN)**：两层线性变换 + ReLU 激活

每个子层后都有 **残差连接（Residual Connection）** 和 **Layer Normalization**：

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

### Feed-Forward Network

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

其中隐层维度通常为 $d_{model} \times 4$，即扩展 4 倍。

## Decoder

每个 Decoder Layer 包含三个子层：

1. **Masked Multi-Head Self-Attention**：对已生成的 token 做自注意力（防止看到未来信息）
2. **Cross-Attention（Encoder-Decoder Attention）**：Query 来自 Decoder，Key/Value 来自 Encoder 输出
3. **Feed-Forward Network**：同 Encoder 的 FFN

Decoder 中的 Mask 确保 position $i$ 的预测只依赖于 $< i$ 的已知输出，实现自回归。

## 位置编码（Positional Encoding）

由于 Transformer 不含递归和卷积结构，无法感知序列中的位置信息，因此需要位置编码来注入位置信号。

原始论文使用正弦/余弦位置编码：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

## 关键组件总结

| 组件 | 说明 |
|:---:|:---:|
| Multi-Head Attention | 多个注意力头并行计算，拼接后线性投影 |
| Scaled Dot-Product Attention | $QK^T / \sqrt{d_k}$ 后 softmax 再乘 V |
| Residual Connection | 残差连接缓解深层网络梯度消失 |
| Layer Normalization | 对每个样本的特征维度做归一化 |
| Positional Encoding | 为无位置感知的注意力注入位置信息 |
| Masked Attention | 防止 Decoder 在训练时看到未来 token |

## Decoder-Only Transformer (LLM 架构)

现代大语言模型（GPT 系列、LLaMA、Gemma 等）大多采用 **Decoder-Only** 架构，即只使用 Transformer 的 Decoder 部分，但去掉 Cross-Attention：

```
Input tokens → [Embedding + Pos Encoding] → Decoder-Only Block × N → LM Head → Output Logits
```

每个 Block 仅包含：
1. **Masked Multi-Head Self-Attention**
2. **Feed-Forward Network**

Decoder-Only 架构的优势：
- 结构简单，训练和推理效率高
- 自回归生成天然适配语言建模
- Scaling Law 效果好
