# Batch Normalization

## paper

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

## 介绍

Batch Normalization (BatchNorm) 是深度学习中最早广泛使用的归一化方法，在 Batch 维度上对特征做归一化。虽然 BatchNorm 在 CNN 中是标配，但在 Transformer 和 LLM 中几乎不使用（LayerNorm/RMSNorm 更适合），了解其原理对于理解归一化方法的发展很重要。

## 公式

对于一个 mini-batch 的输入 $\mathcal{B} = \{x_1, x_2, \ldots, x_B\}$，每个 $x_i$ 是一个特征向量（或 feature map 的某个 channel）：

$$
\mu_{\mathcal{B}} = \frac{1}{B}\sum_{i=1}^{B} x_i
$$

$$
\sigma_{\mathcal{B}}^2 = \frac{1}{B}\sum_{i=1}^{B}(x_i - \mu_{\mathcal{B}})^2
$$

$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}
$$

$$
y_i = \gamma \hat{x}_i + \beta
$$

其中：
- $\mu_{\mathcal{B}}, \sigma_{\mathcal{B}}^2$：在 Batch 维度上计算
- $\gamma, \beta$：可学习的缩放和偏移参数
- $\epsilon$：防止除零（通常 $10^{-5}$）

关键：**归一化是在 batch 维度上对同一个特征位置的不同样本做**，而非对同一个样本的不同特征做。

## 训练与推理的差异

### 训练时

使用当前 mini-batch 的均值和方差：

$$
\mu_{\mathcal{B}} = \frac{1}{B}\sum_{i=1}^{B} x_i, \quad \sigma_{\mathcal{B}}^2 = \frac{1}{B}\sum_{i=1}^{B}(x_i - \mu_{\mathcal{B}})^2
$$

同时维护 running statistics（指数移动平均）：

$$
\mu_{\text{running}} \leftarrow (1 - \alpha) \mu_{\text{running}} + \alpha \mu_{\mathcal{B}}
$$
$$
\sigma^2_{\text{running}} \leftarrow (1 - \alpha) \sigma^2_{\text{running}} + \alpha \sigma^2_{\mathcal{B}}
$$

### 推理时

使用训练时累积的 running statistics，不再依赖 batch：

$$
\hat{x}_i = \frac{x_i - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + \epsilon}}
$$

## CNN 中的 BatchNorm

在卷积层中，BatchNorm 在 $(B, H, W)$ 三个维度上归一化每个 channel：

```
Input: (B, C, H, W)
For each channel c:
  μ_c = mean over (B, H, W) for channel c
  σ²_c = var over (B, H, W) for channel c
  normalize channel c using μ_c and σ²_c
  scale: γ_c * x_norm + β_c
```

可学习参数：$\gamma \in \mathbb{R}^C$, $\beta \in \mathbb{R}^C$（每个 channel 一组）。

## BatchNorm 的问题

1. **依赖 Batch Size**：小 batch size 时，batch 统计量不稳定，影响归一化效果
2. **序列长度不一致**：对变长序列（NLP 任务），batch 内 padding 导致统计量偏差
3. **训练-推理不一致**：训练用 batch 统计量，推理用 running 统计量
4. **分布式训练**：跨设备同步 batch 统计量需要通信（SyncBatchNorm）
5. **自回归模型不适用**：Decode 阶段每次只有 1 个 token，batch 统计量无意义

## 为什么 Transformer 不用 BatchNorm？

| 特性 | BatchNorm | LayerNorm/RMSNorm |
|:---:|:---:|:---:|
| 归一化维度 | Batch | Feature |
| 依赖 batch size | 强 | 无 |
| 变长序列 | 困难 | 自然支持 |
| 自回归推理 | 不适用（batch=1）| 适用 |
| 训练/推理一致 | 不一致（running stats）| 一致 |
| 分布式通信 | 需要 SyncBN | 无需 |

Transformer 和 LLM 选择 LayerNorm/RMSNorm 的根本原因：
- NLP 序列长度不一致，BatchNorm 的 batch 统计量不可靠
- Decode 阶段 batch=1，BatchNorm 退化为无效
- LayerNorm/RMSNorm 在 feature 维度归一化，与 batch 大小无关
