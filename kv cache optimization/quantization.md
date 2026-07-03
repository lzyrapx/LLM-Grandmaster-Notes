# KV Cache Quantization

## 介绍

KV Cache Quantization（KV Cache 量化）通过降低 Key 和 Value 缓存的数据精度来减少显存占用和访存开销。在 Decode 阶段，KV Cache 的读取是内存带宽瓶颈，量化可以显著减少数据传输量。

## 量化格式

| 格式 | 每元素比特 | 相比 FP16 压缩比 | 精度损失 |
|:---:|:---:|:---:|:---:|
| FP16 | 16 bits | 1x | 无 |
| FP8 (E4M3/E5M2) | 8 bits | 2x | 很小 |
| INT8 | 8 bits | 2x | 小 |
| INT4 | 4 bits | 4x | 中等 |
| FP4 / NF4 | 4 bits | 4x | 中等 |

- NF4 在 KV Cache 中的实际局限性
    - NF4（NormalFloat 4）虽然在 QLoRA 中作为**权重**量化格式表现极佳，但在 **KV Cache 的实际部署中几乎无法使用** 。因为 KV Cache 的读取和计算是在自注意力机制（Attention Kernel）中高频发生的，NF4 没有标准的硬件乘法器支持，在 CUDA Kernel 中进行逐元素的非线性去量化开销过大，会严重拖慢推理速度。因此，实际工程中 4-bit KV Cache 几乎全部采用 **INT4** 或 **FP4**。

## 核心挑战

量化 KV Cache 的主要困难在于 **Key 对量化的敏感性高于 Value**：

- **Key 量化**：Key 用于计算注意力分数 $QK^T$，量化误差会放大到注意力权重分布上，导致精度显著下降
- **Value 量化**：Value 用于加权求和，量化误差被平均分散，对精度的影响较小

### K Cache 的异常值问题

研究发现 Key 向量中存在大量异常值（outlier），特别是在较深的 Transformer 层中：

- 少量 channel 的 Key 值极大
- 简单的 per-tensor 量化会导致大量信息丢失
- 需要 per-channel 或分组量化来处理异常值

## 主要方法

### 1. KIVI

[KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)

KIVI 的核心发现：

- **Key 应使用 per-channel 量化**（沿 channel 维度做缩放）
- **Value 应使用 per-token 量化**（沿 token 维度做缩放）

这种非对称量化策略分别适应了 Key 和 Value 的分布特性：

```
K Cache: shape (seq_len, d_k) → per-channel 量化 → 每个 channel 独立的 scale/zp
V Cache: shape (seq_len, d_v) → per-token 量化 → 每个 token 独立的 scale/zp
```

KIVI 在 2-bit 量化下仍能保持大部分模型质量。另外，KIVI 的精度之所以能在 2-bit 下保持稳定，除了非对称量化外，还有一个非常关键的系统级设计—— **保留本地窗口/残差缓存（Residual Cache）** 。KIVI 会将 **最近的 $N$ 个 token**（例如最后 32 个 token）以及 **首个 token（Attention Sink）** 保持在 FP16 精度不进行量化，只有当 token 移出这个滑动窗口后才会被量化为 2-bit。如果不复现这点，全局量化会导致精度崩溃。

### 2. KV Cache 量化 + GQA 配合

GQA 已经减少了 KV 头数，量化可以进一步叠加压缩：

$$
\text{总压缩} = \underbrace{\frac{h}{g}}_{\text{GQA 头数压缩}} \times \underbrace{\frac{16}{b}}_{\text{量化比特压缩}}
$$

例如 GQA (h=64, g=8) + INT4 量化 = 8 × 4 = 32x 压缩。

### 3. SmoothQuant 风格的 KV 量化

类似 SmoothQuant 的思路，将 Key 的异常值从量化困难维度迁移到对量化友好的维度：

$$
\hat{K} = K \cdot \text{diag}(s)^{-1}, \quad \hat{Q} = Q \cdot \text{diag}(s)
$$

将 Key 中 channel 维度的异常值平滑后，可以使 per-tensor 量化也能达到较好效果。

## 实际效果

| 配置 | FP16 基线 | FP8 KV | INT4 KV (KIVI) |
|:---:|:---:|:---:|:---:|
| 显存节省 | 0% | 50% | 75% |
| 精度损失 | 0 | <0.1% | ~1-2% |
| 推理加速 | 基线 | ~1.5x | ~2x |

KV Cache 量化在保持可接受精度损失的前提下，显著降低了推理时的显存占用和访存开销。

- 需要注意的是：
    - 量化 KV Cache 带来的加速并非在所有场景下都有效。在 **Batch Size 较小（如 BS=1）且 Context 较短**时，由于反量化（Dequantization）算子会带来额外的计算开销，量化甚至可能会导致轻微的变慢。
    - 以上加速比主要适用于 **“高并发（Large Batch Size）或长文本（Long Context）等显存访存受限（Memory-Bound）的解码阶段”**

## 其他

#### 基于旋转消除异常值（QuaRot & RotateKV）

既然 Key 中的异常值（Outliers）难以用简单的 per-tensor 量化处理，除了 KIVI 这种复杂的非对称、多 scale 分组量化之外，还有一种非常优雅的思路：**旋转空间**。

* **代表方法**：**QuaRot** (2024)和 **RotateKV**。
* **核心原理**：利用 **Walsh-Hadamard Transform（WHT）** 等正交旋转矩阵乘以输入。由于正交变换具有能量分散的特性，它能将少数 channel 的极大异常值平摊到所有 channel 上，使整个 Tensor 变得非常平滑、无明显异常值。
* **效果**：经过旋转后，Key 和 Value 甚至可以直接使用最简单的 **Per-Tensor INT4/INT2 量化**，而无需复杂的 per-channel 维护，极大地简化了 GPU 算子实现。

## 参考资料

- https://liner.com/review/kivi-tuningfree-asymmetric-2bit-quantization-for-kv-cache
