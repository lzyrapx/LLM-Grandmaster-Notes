# Paged Attention

## paper

[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

## 介绍

Paged Attention 是 vLLM 提出的一种高效 KV Cache 内存管理机制，借鉴了操作系统中虚拟内存和分页（Paging）的思想，解决 LLM 推理中 KV Cache 显存碎片化和浪费的问题。

## 问题：KV Cache 的内存管理挑战

传统 LLM 推理系统中，KV Cache 的管理存在两个主要问题：

1. **预留浪费 (Reservation)**：由于无法预测生成文本的实际长度，传统方法必须按照最大长度（`max_seq_len`）预先分配物理空间。在生成未结束前，这些预留空间一直被闲置（这是最主要的浪费源）。
2. **内部碎片 (Internal Fragmentation)**：分配给请求的最后一个 Block 没有被填满。例如 Block Size 为 16，但请求只占用了其中的 5 个位置，剩下的 11 个位置就是内部碎片。
3. **外部碎片 (External Fragmentation)**：由于传统做法需要为每个请求分配**连续**的显存，而不同请求的生命周期不同，显存频繁申请和释放会导致物理显存出现许多不连续的小空闲块，使后续大请求无法利用。

*注：PagedAttention 彻底消除了 **Reservation** 和 **外部碎片**，并将 **内部碎片** 限制在最后一个 Block 内（平均浪费小于 4%）。*

## 核心思想：分页管理 KV Cache

Paged Attention 将 KV Cache 划分为固定大小的**块（block）**，每个 block 存储固定数量 token 的 K 和 V 向量。这些 block 不需要在物理显存中连续存储：

- **Block Table**：类似操作系统的页表，记录每个请求的逻辑 block 到物理 block 的映射
- **逻辑 block**：请求的 KV Cache 逻辑上由连续的 block 序列组成
- **物理 block**：在 GPU 物理显存中可以非连续存放

```text
逻辑视图（连续）:
  Request A: [Block 0] [Block 1] [Block 2] ...
  Request B: [Block 0] [Block 1] ...

物理视图（非连续）:
  GPU Memory: [A:Block1] [B:Block0] [A:Block0] [A:Block2] [B:Block1] ...
              ↑ 不需要连续存放

Block Table for Request A:
  逻辑 Block 0 → 物理 Block 2
  逻辑 Block 1 → 物理 Block 0
  逻辑 Block 2 → 物理 Block 3
```

## Paged Attention Kernel

在注意力计算时，需要根据 Block Table 找到对应的物理 block 地址来读取 K 和 V：

```text
对于 request 的第 i 个 token:
  逻辑 block_id = i // block_size
  block 内 offset = i % block_size
  物理 block_id = Block_Table[逻辑 block_id]
  K[i], V[i] = Physical_Blocks[物理 block_id][offset]
```

Paged Attention 的 CUDA kernel 需要额外传入 Block Table，通过间接寻址获取 K/V 数据。

## 与 OS 分页的类比

| OS 概念 | Paged Attention 对应概念 |
|:---:|:---:|
| 虚拟地址 | 逻辑 block 序号 |
| 物理地址 | 物理 block 序号 |
| 页表 (Page Table) | Block Table |
| 页 (Page) | Block（固定数量 token 的 KV Cache）|
| 页面置换 | Swap 到 CPU 内存 |
| 共享内存 | Prefix Sharing（共享 system prompt）|

## 关键优势

1. **消除外部碎片**：物理 block 可以非连续存放，无需预分配大块连续内存
2. **减少内部碎片**：block 大小较小（如 16 tokens），浪费最多只有一个 block
3. **动态增长**：KV Cache 按需分配 block，不需要预分配最大长度
4. **Prefix Sharing**：不同请求可以共享相同前缀（如 system prompt）的 block
5. **高效的 Swap**：可以将不活跃的 block 换出到 CPU 内存，需要时换入

## 性能影响

vLLM 实验表明，Paged Attention 相比传统连续 KV Cache 分配：

- GPU 显存利用率接近 100%（几乎零浪费）
- 支持更大的 batch size，吞吐提升 2-4x
- 延迟 P99 更平稳

## 参考资料

- https://medium.com/@danushidk507/pagedattention-with-vllm-i-580b05f3257e
- https://www.hopsworks.ai/dictionary/pagedattention
- https://github.com/VARUN3WARE/Paged-Attention