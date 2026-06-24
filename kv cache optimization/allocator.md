# KV Cache Allocator

## 介绍

KV Cache Allocator 是 LLM 推理引擎中负责管理 KV Cache 显存分配和回收的组件。高效的 KV Cache 分配器直接影响推理吞吐量和延迟，特别是在并发服务多个请求时。

## 问题描述

在 LLM 推理中，每个请求在 Decode 阶段需要不断追加 KV Cache。核心挑战：

1. **预分配 vs 动态增长**：不知道请求的最终序列长度，预分配太多会浪费，预分配太少会溢出
2. **内存碎片**：请求长短不一，频繁分配释放导致外部碎片
3. **多请求并发**：多个请求同时竞争有限的 GPU 显存
4. **Prefill vs Decode 的不同需求**：Prefill 一次性分配大量 KV，Decode 逐步小幅增长

## PagedAttention 的分配器（vLLM）

vLLM 的 PagedAttention 借鉴了操作系统的虚拟内存分页机制：

### 核心概念

- **Block**：固定大小的 KV Cache 单元（如存储 16 个 token 的 KV）
- **Block Table**：每个请求维护的逻辑到物理 block 的映射表
- **物理 Block Pool**：全局的空闲 block 池，所有请求共享

### 分配策略

```
1. 请求到来时，不预分配最大长度的 KV Cache
2. 仅分配当前需要的 block 数量
3. 每生成一个新 token 时，检查当前 block 是否已满
4. block 满时，从物理 block pool 分配新 block
5. 请求完成后，释放所有 block 回物理 pool
```

### 优势

- **按需分配**：只分配实际需要的 block，零浪费
- **消除外部碎片**：block 大小固定，不需要连续物理内存
  - 但它会引入了内部碎片（Internal Fragmentation）。
    - 原因：由于 Block 大小是固定的（例如 16 个 token），如果一个请求在最后一个 Block 中只生成了 1 个 token 随即结束，那么该 Block 剩下的 15 个 token 空间就会被浪费。Block 设得太大，内部碎片会很严重；设得太小，Block Table 的查表开销和内核线程分支分化（Warp Divergence）会上升。
- **共享 block**：不同请求的相同前缀（如 system prompt）可以共享 block
- **最大的 batch size**：显存利用率接近 100%

## Continuous Batching 的分配器配合

在 Continuous Batching 中，请求可以随时到达和离开：

- **请求到达**：分配初始 block，开始 prefill
- **请求完成**： 在 SGLang、vLLM 中，请求结束时，其对应的 Block 不会被立即销毁或彻底释放回物理池。
  - 为了支持 Prefix Caching（前缀缓存），已结束请求的 Block 会被保留，并放入一个类似于 LRU（Least Recently Used） 的缓存队列或树状结构中。如果有新的请求带有相同的 Prefix（如相同的 System Prompt 或多轮对话历史），引擎可以直接命中并复用这些 Block，免去重复 Prefill 的计算。只有当全局空闲显存彻底枯竭时，分配器才会按照 LRU 策略真正释放这些 Block。
- **请求抢占（Preemption）**：当显存不足时，暂停低优先级请求，释放其 block 给高优先级请求

## Orca 风格的分配器

Orca 提出 iteration-level scheduling（即 Continuous Batching），每个 iteration 重新决定哪些请求参与计算：

```
每个 decoding step:
  1. 检查完成请求，释放 block
  2. 检查新到达请求，分配 block 开始 prefill
  3. 如果显存不足，执行抢占策略
  4. 构建 batch 执行一步前向
```

但在 KV Cache 内存管理上，Orca 恰恰是非常保守且低效的。

- Orca 采用的是预分配最大长度（Max-length pre-allocation）（或基于最大输出长度的静态预留）。它在请求开始时，必须在显存中预留该请求所需的最大 Token 数量的连续空间，导致其显存浪费极其严重。
- 正因为 Orca 的这种静态内存分配方式成为了并发瓶颈，vLLM 团队才在 SOSP '23 提出了 PagedAttention，将“Continuous Batching（来自 Orca）”与“虚拟内存分页管理（PagedAttention）”结合起来。因此，不宜将高效动态的块分配器称为“Orca 风格”，Orca 仅代表了迭代级调度。

## 其他分配策略

### 1. 预分配最大长度

最简单的策略，为每个请求预分配最大序列长度的连续 KV Cache：

- 优点：实现简单，无分配延迟
- 缺点：严重浪费显存（实际使用率可能 < 50%）

### 2. Slab 分配

类似操作系统的 Slab Allocator，预分配不同大小的 slab：

- 小 slab: 短序列请求
- 大 slab: 长序列请求
- 减少碎片，但需要预估长度分布

### 3. Swap to CPU

当 GPU 显存不足时，将不活跃请求的 KV Cache 换出到 CPU 内存：

- 需要 GPU-CPU 高带宽传输（如 PCIe/NVLink）
- 换入换出延迟需要与计算重叠
- vLLM 实现了 block-level 的 CPU swap

## 其他

### RadixAttention（基数树缓存）

SGLang 提出了 RadixAttention，将 KV Cache 分配与 **基数树（Radix Tree）** 数据结构深度结合：

原理：

- 将 Token 序列作为路径，将 KV Cache Block 作为树的节点。

优势：

- 任意前缀共享：不限于静态的 System Prompt，它支持多轮对话历史、Few-shot 示例、RAG（检索增强生成）中的参考文档等任意位置的重合前缀。

- 动态驱逐：当显存不足时，分配器会像操作系统回收页面一样，在基数树上利用 LRU 算法选择性地驱逐叶子节点（最近最少使用的分支），保留高频使用的根节点和公共前缀。

### vAttention

基于硬件 native 虚拟内存的分配器（免去 PagedAttention 的内核重写）
PagedAttention 虽好，但它有一个巨大的系统代价：

- 由于物理内存不连续，迫使开发者必须重写 Attention Kernel（无法直接使用原生、优化到极致的 FlashAttention）。这带来了显著的软件复杂度和查表延迟。

为此，微软等团队提出了 vAttention：

核心思想：

- 利用 CUDA 驱动级的虚拟内存管理 API（CUDA VMM API，如 cuMemCreate、cuMemMap）。

机制：

- 在虚拟内存地址空间中，为每个请求保留一块连续的大区域（支持原生 FlashAttention 直接寻址）；但在物理显存中，分配器以物理页为单位（如 2MB 或 64KB），在运行时按需动态映射物理显存。

结果：

- 既实现了 PagedAttention 的动态按需分配，又保留了 KV Cache 的虚拟地址连续性，使开发者能够直接使用未修改的原生高性能 Attention 内核。

### Chunked Prefill（分块 Prefill）对分配器的影响

在长文本场景下，一个超长的 Prefill 请求会长时间独占 GPU 计算资源，导致其他 Decode 请求严重卡顿（称为 TTFT 与 ITL 的冲突）。

Chunked Prefill 将一个大 Prefill 请求拆分为多个固定大小的 Chunk（例如每次只计算 512 个 token 的 Prefill），并将其与 Decode 请求混合进同一个 Batch。

分配器的配合：

- 分配器不能再采用“一次性为整个 Prompt 分配全部空间”的简单策略，而必须支持增量式、跨 Iteration 的物理 Block 动态追加，同时在多次迭代间妥善维护中间计算状态。

### Prefill-Decode Disaggregation 下的分布式分配器

由于 Prefill（计算密集型）和 Decode（访存密集型）对硬件资源的要求截然不同，当前业界主流架构正朝着 **PD 分离式推理** 发展：

流程：

- Prefill 在专用的 Prefill 节点上运行，完成后，将生成的 KV Cache 通过高速网络（如 NVLink 或 RDMA）传输到 Decode 节点上继续生成。

分配器的挑战：

- 单机分配器演变为分布式分配器（如 LMCache, NIXL 项目）。分配器需要具备远程内存管理、网络传输缓存（Network-aware caching）以及跨节点内存块地址对齐和重定位的能力。