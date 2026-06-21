# Layer Normalization

## paper

[Layer Normalization](https://arxiv.org/abs/1607.06450)

## 介绍

Layer Normalization (LayerNorm) 对单个样本的所有特征维度做归一化，使其均值为 0、方差为 1，然后通过可学习的缩放和偏移参数恢复表达力。LayerNorm 是 Transformer 中的标准归一化方法。

## 公式

给定输入 $\mathbf{x} = [x_1, x_2, \ldots, x_d]$：

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i
$$

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

其中：
- $\mu, \sigma^2$：在特征维度 $d$ 上计算
- $\epsilon$：防止除零的小常数（通常 $10^{-5}$）
- $\gamma, \beta$：可学习的缩放和偏移参数，形状 $(d,)$

## 计算图

```
Input: x (shape: [B, S, d])
       ↓
  Mean over d: μ (shape: [B, S, 1])
       ↓
  Variance over d: σ² (shape: [B, S, 1])
       ↓
  Normalize: x̂ = (x - μ) / √(σ² + ε)
       ↓
  Scale & Shift: y = γ * x̂ + β
```

## 与 Batch Norm 的对比

| 特性 | Batch Norm | Layer Norm |
|:---:|:---:|:---:|
| 归一化维度 | Batch 维度 | Feature 维度 |
| 统计量 | 不同样本同特征 | 同一样本所有特征 |
| 依赖 batch size | 是 | 否 |
| 序列模型适用性 | 差（变长序列） | 好 |
| 推理行为 | 需要running mean/var | 与训练一致 |
| 常用场景 | CNN | Transformer, RNN |

## LayerNorm 在 Transformer 中的位置

### Pre-Norm vs Post-Norm

**Post-Norm**（原始 Transformer）：

$$
x_{l+1} = \text{LayerNorm}(x_l + \text{Attn}(x_l))
$$

**Pre-Norm**（现代 LLM 常用）：

$$
x_{l+1} = x_l + \text{Attn}(\text{LayerNorm}(x_l))
$$

Pre-Norm 的优势：
- 训练更稳定，不需要 learning rate warmup
- 梯度可以直接通过残差连接流过，缓解梯度消失
- 大部分现代 LLM（GPT, LLaMA, Gemma）使用 Pre-Norm

## PyTorch 实现

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x shape: (B, S, d)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

## CUDA 实现

```C++
// LayerNorm kernel (简化版)
__global__ void layernorm_kernel(float* out, const float* x, const float* gamma, const float* beta,
                                 int d, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 1. 计算均值 (Mean)
    float sum = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        sum += x[row * d + i];
    }
    // block_reduce_sum 可以基于 Warp Shuffle + 共享内存自己实现
    // 或者使用 cub 库（生产环境）
    sum = block_reduce_sum(sum);
    float mean = sum / d;

    // 2. 计算方差和标准差 (Variance & Standard Deviation)
    float sum_sq_diff = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float diff = x[row * d + i] - mean;
        sum_sq_diff += diff * diff;
    }
     // block_reduce_sum 可以基于 Warp Shuffle + 共享内存自己实现
    // 或者使用 cub 库（生产环境）
    sum_sq_diff = block_reduce_sum(sum_sq_diff);
    float std = sqrtf(sum_sq_diff / d + eps);

    // 3. 归一化 + 缩放 + 偏移
    for (int i = tid; i < d; i += blockDim.x) {
        out[row * d + i] = gamma[i] * (x[row * d + i] - mean) / std + beta[i];
    }
}
```

## 计算开销

LayerNorm 的主要开销：
- 两次规约操作（mean 和 variance），在特征维度 $d$ 上
- 在 GPU 上规约操作在 $d$ 较大时效率较低
- $\gamma$ 和 $\beta$ 的逐元素乘法开销较小
