
### Cooperative 和 Ping-pong 定义

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

在 NVIDIA GPU 上，对于 GEMM 类的 Compute-bound 算子，性能优化的目标通常是需要程序可以持续地、饱和地利用所有 SM Core 上的 Tensor Core 运算单元。在 CUTLASS GEMM Kernel 中，Mainloop 阶段主要利用的是 Tensor Core 运算单元，而 Epilogue 阶段则是完成一些额外的计算操作（例如实施激活函数）并将结果写回 Global Memory，这些操作只依赖于 Cuda Core，不依赖于 Tensor Core。因此，结合性能优化的目标，我们希望在整个 Kernel 的生命周期中尽可能的使用 Mainloop 掩盖 Epilogue 的开销，避免将 Epilogue 直接暴露在 Timeline 上，以最大化 Tensor Core 的利用率。


### 总结

在 H100 环境下压测过，跑大部分 shape 场景，Ping-pong scheduler 会比 Cooperative scheduler 稍微快一点。

主要原因：

- Ping-pong 调度策略通过 Ordered Sequence Barrier 严格的约束了两个 Consumer Warp Group 的执行顺序，让两个 Consumer Warp Group 交错执行 Mainloop 和 Epilogue，有效的 Overlap 掉了 Epilogue 的开销。
- Cooperative 中的两个 Consumer Warp Group 依赖于同样的数据，在数据到达后以一种"竞争"的模式使用 Tensor Core 计算资源，在“势均力敌”的情况下，Mainloop 的执行结束时间相接近，导致 Epilogue 不能够被有效的 Overlap。
