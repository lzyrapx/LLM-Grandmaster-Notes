### SmoothQuant

- paper: https://arxiv.org/abs/2211.10438
- code: https://github.com/mit-han-lab/smoothquant

#### 背景

[LLM.int8()](https://arxiv.org/pdf/2208.07339) 论文里提出：

- activation 比 weight 更难量化，后者数据分布一般比较均匀。
- outlier 的存在会导致非 outlier 值在 per-tensor 级别的量化误差大。
- activation 的量化方式不适宜用 per-channel，weight 用 per-tensor 或者 per-channel 都可以，这是因为 activation 用 per-channel 的话很难 dequantize。
- LLM.int8() 使用混合精度推理，每次计算都会恢复异常值至 FP16，但这需要在运行进行异常值 detecting，scattering 和 gathering，计算非常慢。
- 研究发现 weight 分布是统一且平滑的，**使用 INT8 甚至 INT4 量化 LLM 的权重不会降低准确性**。

针对以上情况，提出 SmoothQuant 来解决量化问题。

#### 实现