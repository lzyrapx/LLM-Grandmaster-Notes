## Cooperative 和 Ping-pong 定义

Cooperative 和 Ping-pong 都是 Warp Specialization Persistent Kernel 的调度策略。

关于 Warp Specialization 这些调度策略 以及 Persistent 的概念。可以参考以下文档：

- <https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md>
- <https://github.com/NVIDIA/cutlass/issues/2181>
- [Deep Dive on CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)

Cooperative 调度策略 和 Ping-pong 调度策略的区别：

- Cooperative 调度策略会让两个 Consumer Warp Group 计算同一个 Output Tile，一个 Warp Group 计算 Output Tile 的上半部分，另一个 Warp Group 计算 Output Tile 的下半部分。
- Ping-pong 调度策略，两个 Consumer Warp Group 则是分别计算不同的 Output Tile。

因此，如果为这两种调度策略设置同样大小的 Tile Shape，Ping-pong 调度策略的 Register Pressure 一定更大，因为它需要单个 Consumer Warp Group 来为整个 Output Tile 分配寄存器资源用于存储运算结果，而对于 Cooperative，单个 Consumer Warp Group 仅需为一半的 Output Tile 来分配寄存器。因此在很多 CUTLASS Examples 中，Ping-pong 的 Tile Shape 通常都是 Cooperative 的一半。

比如，[57_hopper_grouped_gemm](https://github.com/NVIDIA/cutlass/blob/main/examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu)

```cpp
// Different configs for pingpong/cooperative
struct CooperativeConfig {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using TileShape           = Shape<_256,_128,_128>;
  using ClusterShape        = Shape<_1,_2,_1>;
};

struct PingpongConfig {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape           = Shape<_128,_128,_128>;
  using ClusterShape        = Shape<_2,_1,_1>;
};
```

显然，Ping-pong 调度策略的 Output Tile 仅有 Cooperative 调度策略的一半大小。

在 NVIDIA GPU 上，对于 GEMM 类的 Compute-bound 算子，性能优化的目标通常是**需要程序可以持续地、饱和地利用所有 SM Core 上的 Tensor Core 运算单元**。在 CUTLASS GEMM Kernel 中，Mainloop 阶段主要利用的是 Tensor Core 运算单元，而 Epilogue 阶段则是完成一些额外的计算操作（例如实施激活函数）并将结果写回 Global Memory，这些操作只依赖于 Cuda Core，不依赖于 Tensor Core。因此，结合性能优化的目标，希望在整个 Kernel 的生命周期中尽可能的使用 Mainloop 掩盖 Epilogue 的开销，避免将 Epilogue 直接暴露在 Timeline 上，以最大化 Tensor Core 的利用率。


视频：[CUTLASS 2.x 与 3.x 的入门使用](https://www.bilibili.com/video/BV1XH4y1c7JZ?spm_id_from=333.788.videopod.sections&vd_source=3187e54ee4327cdd9b00a232b8ccb71c)

![cooperative](../../assets/cooperative_gemm.png)

从上图的 cooperative 示例可以看到，灰色部分代表 Epilogue， 可以看到使用 Cooperative 调度策略时，Epilogue 部分是暴露在 Timeline 上的。

相比之下，使用 Ping-pong 调度策略时，Epilogue 部分则是完全被 Mainloop 部分 Overlap 掉了，如下图所示：

![ping-pong](../../assets/ping-pong_gemm.png)

上面的两张图片只是理想情况下的简单示例。接下来看一看真实 Kernel 的 Timeline 是否和上图具有一致的现象，此时我们需要利用 nc u的一个新特性——[PM Sampling](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#pm-sampling)。

按照如下配置分别运行使用 Cooperative 调度策略和 Ping-pong 调度策略的 FP8 Blockwise Scaling Grouped GEMM：

```bash
num_groups=256
M=128
N=512
K=7168
```

观察 Cooperative Kernel 的 PM Sampling Timeline：

![Cooperative_PM_Sampling](../../assets/Cooperative_PM_Sampling.png)

可以看到，Tensor Pipe Throughput 出现了非常明显的周期性下降的现象，在 Timeline 上形成了一些“缺口”。如果我们仔细的去数一数这些“缺口”的数量，可以发现这些“缺口”共有 12 个。因为使用的是 H20 GPU（具有 78 个 SM Core），并且 Cooperative 调度策略使用的 Tile Shape 为 (128, 128, 128)，因此我们可以推算每个 CTA 需要计算的 Output Tile 的数量：

```bash
>>> num_groups = 256 
>>> M = 128
>>> N = 512
>> TileShapeM = 128 
>> TileShapeN 128 
>>> CTAs = 78
>> num_groups * ((M / TileShapeM) * (N / TileShapeN) / CTAs 
13.128205128205128
```

显然，多数 CTA 需要计算 13 个 Output Tile，在 Warp Specialization Persistent Kernel 中，每个 SM Core 只调度一个 CTA，因此 SM Core 上的运行状况就是单个 CTA 的运行状况。12 个缺口恰好对应了前 12 个 Output Tile 的 Epilogue 阶段。相比之下，Ping-pong 调度策略的 PM Sampling Timeline 显示 Tensor Pipe Throughput 始终处于一个相对稳定的水平，不会出现明显的“缺口”现象：

![Pingpong_PM_Sampling](../../assets/Pingpong_PM_Sampling.png)

## 总结

在 H100 环境下压测过，跑大部分 shape 场景，Ping-pong scheduler 会比 Cooperative scheduler 稍微快一点。

主要原因：

- Ping-pong 调度策略通过 Ordered Sequence Barrier 严格的约束了两个 Consumer Warp Group 的执行顺序，让两个 Consumer Warp Group 交错执行 Mainloop 和 Epilogue，有效的 Overlap 掉了 Epilogue 的开销。
- Cooperative 中的两个 Consumer Warp Group 依赖于同样的数据，在数据到达后以一种"竞争"的模式使用 Tensor Core 计算资源，在“势均力敌”的情况下，Mainloop 的执行结束时间相接近，导致 Epilogue 不能够被有效的 Overlap。
