# RMS Normalization

## paper

[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)

## 介绍

RMS Normalization (RMSNorm) 是 LayerNorm 的简化版本，去掉了均值平移（mean centering）步骤，只使用 RMS（均方根）做归一化。RMSNorm 在现代 LLM 中广泛应用（LLaMA, Gemma, Mistral 等），因其计算效率更高且效果与 LayerNorm 相当。

## 公式

给定输入 $\mathbf{x} = [x_1, x_2, \ldots, x_d]$：

$$
\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}
$$

$$
\hat{x}_i = \frac{x_i}{\text{RMS}(\mathbf{x}) + \epsilon}  \quad \text{或} \quad \hat{x}_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}
$$

$$
y_i = \gamma_i \hat{x}_i
$$

与 LayerNorm 的区别：
- **无均值平移**：不计算 $x_i - \mu$，直接 $x_i / \text{RMS}$
- **无偏移参数**：通常不使用 $\beta$（部分实现保留）
- 只有 $\gamma$ 一个可学习参数

## LayerNorm vs RMSNorm

| 方面 | LayerNorm | RMSNorm |
|:---:|:---:|:---:|
| 均值平移 | $x_i - \mu$ | 无 |
| 归一化因子 | $\sqrt{\sigma^2 + \epsilon}$ | $\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$ |
| 可学习参数 | $\gamma, \beta$ | $\gamma$（或 $\gamma, \beta$）|
| 计算量 | 较高（需计算均值和方差）| 较低（只需 RMS）|
| 效果 | 基准 | 相当或略优 |
| 常用模型 | GPT-2/3, BERT | LLaMA, Gemma, Mistral |

## 为什么 RMSNorm 有效？

1. **Re-centering 不必要**：LayerNorm 的均值平移（re-centering）在大多数情况下效果有限，因为残差连接已经隐式地维持了零均值
2. **Re-scaling 是关键**：归一化的主要作用是 re-scaling（调整激活值的尺度），RMSNorm 保留了这一点
3. **计算效率**：省去均值计算和偏移参数，规约操作从两次（mean + var）减少到一次（RMS）

## 数学等价性

当输入 $\mathbf{x}$ 的均值为 0 时：

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2 = \frac{1}{d}\sum_{i=1}^{d}x_i^2 - \mu^2 = \frac{1}{d}\sum_{i=1}^{d}x_i^2 = \text{RMS}(\mathbf{x})^2
$$

此时 LayerNorm 和 RMSNorm 完全等价。

## PyTorch 实现

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # x shape: (B, S, d)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm

# PyTorch 内置版本 (PyTorch 2.4+)
# nn.RMSNorm(d_model, eps=1e-5)
```

## CUDA 优化

RMSNorm 的 CUDA 实现比 LayerNorm 简单：

```C++
// RMSNorm kernel (简化版)
__global__ void rmsnorm_kernel(float* out, const float* x, const float* gamma,
                                int d, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 计算 RMS
    float sum_sq = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float val = x[row * d + i];
        sum_sq += val * val;
    }
    // block_reduce_sum 可以基于 Warp Shuffle + 共享内存自己实现
    // 或者使用 cub 库（生产环境）
    sum_sq = block_reduce_sum(sum_sq);
    float rms = sqrtf(sum_sq / d + eps);

    // 归一化 + 缩放
    for (int i = tid; i < d; i += blockDim.x) {
        out[row * d + i] = gamma[i] * x[row * d + i] / rms;
    }
}
```

相比 LayerNorm，少了一次规约（无需计算均值）和一次减法。
