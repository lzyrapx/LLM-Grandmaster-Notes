
##  SageAttention 系列：从 8-bit 到 FP4，Attention 加速的演进之路

在长上下文大语言模型（LLMs）和高分辨率视频生成模型（如 CogVideoX、HunyuanVideo、Wan 2.1）中，Attention 机制的 $O(N^2)$ 计算复杂度常常成为推理和训练的绝对瓶颈。尽管业内对线性层（Linear Layer）的量化（如 W4A8、W8A8）已经非常成熟，但对 Attention 内部的 $Q, K, V$ 进行极低比特量化却因**严重的异常值（Outliers）问题**和**累加精度丢失**而停滞不前。

**SageAttention 系列** 通过一系列精妙的数学平滑技巧和底层硬件级优化，实现了“即插即用（Plug-and-play）”的无损加速，最终还实现了 Blackwell 架构下的 FP4 极低精度推理和 8-bit 训练加速。

## 背景

标准的 Attention 计算公式如下：

$$ P = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) $$
$$ O = P V $$

如果我们想用低精度 Tensor Cores（如 INT8/INT4/FP8/FP4）来加速这两个主要的矩阵乘法（Matmul），会面临以下痛点：
1. **通道异常值（Channel-wise Outliers）**： $K$ 和 $V$ 的某些通道存在极大值，如果直接进行常规量化，会导致有效量化位宽大幅缩水，引发严重精度损失。
2. **非线性激活**：Softmax 的存在导致 $P$ 的数值分布动态范围极大。
3. **累加器溢出与截断**：极低精度（如 FP8/FP4）矩阵乘法在底层累加时，极易发生精度截断。

## Sage Attention 1

### paper

https://arxiv.org/pdf/2410.02367v7

### 思路

第一代 SageAttention 证明了对 Attention 内部进行 8-bit 量化是完全可行且无损的。
1. **通道级平滑（Channel-wise Smoothing）**：为了解决 $K$ 矩阵的异常值，SageAttention 引入了平滑矩阵 $S$（一个对角阵），使得：

$$ QK^\top = Q(S^{-1}S)K^\top = (QS^{-1})(KS)^\top = \bar{Q}\bar{K}^\top$$

通过这种等价代换， $\bar{K}$ 的异常值被抹平，从而可以安全地对 $\bar{Q}$ 和 $\bar{K}$ 进行 **INT8** 量化。

2. **混合精度计算**： $QK^\top$ 采用 INT8 Matmul，而 $PV$ 的计算则使用 **FP16 Matmul + FP16 Accumulator**，从而在 RTX 3090/4090 上实现了比 FlashAttention2 快 2.1 倍的性能，且端到端指标几乎零损失。

### 数学公式总结 (核心平滑机制)

以对 K 进行通道级平滑为例，假定量化函数为 $Quant(X) = \text{round}(X / s) \cdot s$。
由于 $K$ 在某些隐藏维度（Head Dimension）经常出现数值极点 $k_{max}$：
我们定义对角缩放矩阵：

$$ S = \text{diag}(\max(|K_{:,:,c}|))^\alpha $$

其中 $\alpha$ 是一个调节强度超参数。应用平滑后：

$$ Q_{smooth} = Q \cdot S^{-1} $$

$$ K_{smooth} = K \cdot S $$

这样 $K$ 被压制了极值， $Q$ 则反向放大了对应维度，使得内积不变，但两个矩阵的数值分布极大地适配了 INT8 或 INT4 的量化位阶分布（Quantization Bins）。

## Sage Attention 2

### paper

https://arxiv.org/abs/2411.10958

### 思路

既然 8-bit 成功了，能不能上更猛的 4-bit（INT4）硬件指令？

1. **线程级（Per-thread / Warp-level）INT4 量化**：为了缓解 INT4 带来的极端量化误差，SageAttention 2 放弃了粗粒度，转向了对硬件亲和的“线程级/Warp级”极细粒度量化，让 $Q$ 和 $K$ 在极小范围内共享 Scale。
2. **彻底的异常值平滑**：除了平滑 $K$，SageAttention 2 提出了同时对 $Q$ 甚至 $V$ 进行平滑处理的方法。
3. **FP8 引入与两级累加策略（Two-level Accumulation）**：
   将 $P$ 和 $V$ 量化为 **FP8**。为了防止 FP8 MMA/WGMMA 指令在内部发生精度截断，作者巧妙设计了在共享内存（Shared Memory）和寄存器中进行**FP32/FP16 的两级缓冲累加（Buffering）**。这使得速度狂飙的同时，规避了 FP8 累加带来的灾难性画质崩坏。
   最终在 RTX 4090 上比 FlashAttention-2 快约 3 倍，在 Hopper 架构上速度可以比肩 FlashAttention-3 但精度碾压后者。

## Sage Attention 2++

这并非一篇新论文，而是开源社区中的一次重大工程迭代（v2.2.0 版本发布）。
**核心贡献**：
- 在维持 SageAttention 2 相同数学精度的前提下，对 Ampere, Ada 到 Hopper 的底层 Kernel 进行了极致的手工调优。
- 支持了非 CUDA Graph 模式下的 `torch.compile`（Torch 编译融合），极大地扩展了工程兼容性。
- 相比 v2.0，在 FP16 与 FP8 模式下分别榨取了额外 10%~20% 的运行速度，被广泛集成到 Stable Diffusion 与 ComfyUI 社区用于加速视频生成大模型（如 Wan 2.1）。

## Sage Attention 3

### paper

https://arxiv.org/abs/2505.11594

### 思路

随着 RTX 5090 / B200（Blackwell 架构）的发布，硬件原生支持了 FP4 Tensor Cores。

1. **微缩放 FP4 推理（Microscaling FP4 Attention）**：引入对 $P$ 矩阵的两级缩放（Two-level Scaling）策略，将 $P, V$ 和 $Q, K$ 压缩至 **FP4**。在 RTX 5090 上达到了恐怖的 1038 TOPS（是该卡上最快 FlashAttention 速度的 5 倍）。
2. SageAttention 3 将 8-bit 量化引入了 Attention 的**前向和反向传播（Forward & Backward）**。实验证明在微调（Fine-tuning）任务中完全无损，但从头预训练的话收敛稍慢。

## 代码

可以直接使用它来替换 PyTorch 的 `F.scaled_dot_product_attention`。

**安装**：
```bash
# 推荐环境: Python 3.11+, PyTorch 2.4.0+, Triton nightly
pip install sageattention
```

```python
import torch
from sageattention import sageattn

# 模拟输入参数: (batch_size, head_num, seq_len, head_dim)
batch_size, head_num, seq_len, head_dim = 2, 32, 4096, 128
q = torch.randn(batch_size, head_num, seq_len, head_dim, dtype=torch.float16, device="cuda")
k = torch.randn(batch_size, head_num, seq_len, head_dim, dtype=torch.float16, device="cuda")
v = torch.randn(batch_size, head_num, seq_len, head_dim, dtype=torch.float16, device="cuda")

# 直接调用 SageAttention API
# tensor_layout "HND" 代表 (Batch, Head, Seq_len, Dim)
# is_causal 控制是否使用因果掩码（自回归LLM设为True，图像/视频生成设为False）
attn_output = sageattn(
    q, k, v, 
    tensor_layout="HND", 
    is_causal=False, 
    smooth_k=True # 启用 K 的平滑
)

print(attn_output.shape) # 输出: torch.Size([2, 32, 4096, 128])
```

## 总结
1. 面向 RTX 消费卡的 Attention 优化
2. SageAttention 1：8-bit Attention
3. SageAttention 2/2++：INT4 与 FP8 的混合收益
4. SageAttention 3：RTX Blackwell FP4 收益，支持 fwd + bwd
5. 日常跑图像/视频生成用 SageAttention 2++
6. 50 系显卡用 SageAttention 3

