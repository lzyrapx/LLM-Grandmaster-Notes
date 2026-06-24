# KV Cache Share

## 介绍

KV Cache Share（KV Cache 共享）是 LLM 推理中的优化技术，允许多个请求共享相同的 KV Cache 部分，从而减少冗余的显存占用和重复计算。

## 问题：KV Cache 冗余

在实际 LLM 服务中，大量请求共享相同的 prompt 前缀：

```
Request A: [System Prompt] + [User Query A]
Request B: [System Prompt] + [User Query B]
Request C: [System Prompt] + [User Query C]
```

每个请求的 System Prompt 部分的 KV Cache 完全相同，但传统方案为每个请求独立缓存，造成 $N$ 倍冗余。

## 共享策略

### 1. Prefix Caching（前缀缓存）

最直接的共享策略：识别相同的前缀，只计算和缓存一次。

```
物理 Block Pool:
  Block 0: [System Prompt tokens 0-15]   ← 被请求 A, B, C 共享
  Block 1: [System Prompt tokens 16-31]  ← 被请求 A, B, C 共享
  Block 2: [User Query A tokens 0-15]    ← 仅请求 A
  Block 3: [User Query B tokens 0-15]    ← 仅请求 B
  Block 4: [User Query C tokens 0-15]    ← 仅请求 C

Block Table:
  Request A: [0] → [1] → [2] → ...
  Request B: [0] → [1] → [3] → ...
  Request C: [0] → [1] → [4] → ...
```

前缀的 KV Cache 通过引用计数（Reference Counting）管理：

- 新请求匹配到已有前缀时，引用计数 +1
- 请求完成时，引用计数 -1
- 引用计数为 0 的 block 不会被立即物理释放（清空或抹除）
  - 机制：当一个请求结束，其占用的 block 引用计数归 0 后，这些 block 会被移动到一个 LRU（Least Recently Used，最近最少使用）队列中。它们的数据仍保留在显存中。
  - 作用：如果后续有新请求刚好匹配到这部分前缀，引擎可以立刻将这些 block 从 LRU 队列中“拯救”出来并重新复用（即 Cache Hit）。只有当显存空间不足，需要分配给新激活的请求时，引擎才会真正按照 LRU 策略驱逐（Evict）并覆盖这些引用计数为 0 的块。如果一归 0 就立即释放，串行（前后相继）的相同前缀请求将无法享受缓存带来的收益。

### 2. RadixAttention (SGLang)

#### paper

[SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)

#### 思路

SGLang 的 RadixAttention 使用 Radix Tree 来高效管理 KV Cache 的共享：

- **Radix Tree**：存储 token 序列到物理 KV block 的映射
- **自动前缀匹配**：新请求到来时，在 Radix Tree 中查找最长匹配前缀
- **动态拆分与合并**：前缀不匹配时，自动拆分节点
  - 在 Radix Tree 的结构定义中，不可能存在具有相同前缀的多个独立叶子节点，因为前缀树在插入时天生就保证了相同前缀会合流到同一条路径上
  - Radix Tree 中的“合并”（通常称为路径压缩或节点合并）发生在删除或驱逐过程中。当某个分支被驱逐后，如果一个父节点只剩下一个子节点，Radix Tree 会将这个父节点与子节点垂直合并成一个节点，用来节省树节点的开销并提高检索效率。

```
Radix Tree:
  root
  ├── "You are a helpful"  → Block[0, 1]
  │   ├── "assistant"      → Block[2]     (Request A, B 共享)
  │   │   ├── "Answer:"    → Block[3]     (Request A)
  │   │   └── "Explain:"   → Block[4]     (Request B)
  │   └── "coder"          → Block[5]     (Request C)
```

**优势**：

- 支持任意前缀匹配（不仅是固定 system prompt）
- 自动发现共享机会
- O(序列长度) 的查找效率

### 3. Chunk-level 共享

前缀匹配的最小粒度为 chunk（如 PagedAttention 中的一个 block，16 tokens）：

- 前缀必须在 chunk 边界上匹配才能共享
- 不对齐到 chunk 边界的前缀无法共享（需要重新计算少量 token 以对齐）
- 应用层可以通过在 system prompt 末尾填充到 chunk 边界来优化共享率

实际上 Chunk-level（块级）和 Token-level（Token 级）是两种不同的逻辑匹配粒度。

- **vLLM 的 Prefix Caching（块级哈希）**：采用固定大小的块（如 16 或 32 tokens）进行哈希映射。匹配必须严格对齐到块边界、。如果 System Prompt 是 35 个 token，块大小是 16，那么前 32 个 token（2 个整块）可以被共享，而最后 3 个 token 即使后续请求完全相同，也会因为无法填满一个整块而在每次 prefill 时重新计算。
- **SGLang 的 RadixAttention（Token 级逻辑）**：虽然底层的物理显存分配仍然是块状的（Paged），但它在**逻辑结构（Radix Tree）上实现了 Token 级的精细匹配**。对于未对齐到块边界的剩余 token，SGLang 能够精确定位到未对齐的分支点，只对不匹配的部分以及未满块的尾部进行少量的重计算（prefill），最大化了共享比例。

## 显存节省

对于 $N$ 个请求共享长度为 $P$ 的前缀：

| 方案 | KV Cache 总量 |
|:---:|:---:|
| 无共享 | N * P * per_token_size |
| Prefix Caching | P * per_token_size |
| 节省 | (N - 1) * P * per_token_size |

当 $N$ 很大（如高并发场景）且 $P$ 很长（如长 system prompt），节省非常显著。

## 额外优化：Prompt 预计算

共享前缀的 KV Cache 可以预先计算并长期缓存：

1. **冷启动**：Service 启动时预先计算常用 system prompt 的 KV Cache
2. **热缓存**：最常用的前缀常驻 GPU 显存
3. **温缓存**：较不常用的前缀换出到 CPU 内存，需要时换入

## 其他

- Prefill 阶段的算力节省与 TTFT 的关系
  - KV Cache 共享不仅降低了显存占用（空间维度），能显著地降低了计算延迟（时间维度）
    -  在 LLM 推理中，首次生成 Token 的阶段（Prefill 阶段）具有 $O(L^2)$ 的计算复杂度（$L$ 为输入序列长度）
    - 共享前缀被缓存后，后续请求可以直接**跳过被缓存部分的 Prefill 计算**，直接进入 Decode 阶段（或者只对未缓存的 Query 部分进行 Prefill）
    - 可以使**首字延迟（TTFT, Time-to-First-Token）**从数秒级降低至毫秒级，大大提升了长文本（如 RAG、长 System Prompt）场景下的用户体验
- 动态多分支任务（Agent 与 Tree-of-Thought）
  - RadixAttention 最强大的场景不仅仅是共享 System Prompt，而是**多轮对话（Multi-turn Chat）**和**智能体（Agent）工作流**
    - 在 **Tree-of-Thought（思维树）** 或 **Self-Consistency（自一致性采样）** 算法中，模型需要从同一个上下文分支出多个不同的生成路径
    - Radix Tree 结构天然契合这种“分叉”模式，多个并行分支（Branch）可以共享它们分叉前的所有历史 KV Cache，无需为每个分支复制一份完整的历史上下文
- 分级缓存（Hierarchical KV Cache / Offloading）
  - GPU 显存（HBM）极为昂贵且容量有限，无法无限存储历史前缀。因此，业界引入了分级缓存（如 LMCache 或 vLLM/SGLang 的 CPU Offloading）
    - **热缓存（GPU HBM）**：存放当前活跃请求和极高频的前缀。
    - **温缓存（CPU Host Memory）**：当 GPU 显存紧张时，将引用计数为 0 的 LRU 块换出（Swap out）到系统内存（DDR）中。当新请求命中该前缀时，再快速换入（Swap in）到 GPU，避免重新计算。
    - **冷缓存（本地磁盘 SSD / 分布式存储）**：将极低频但体量庞大的知识库前缀 KV 序列持久化在磁盘或远程对象存储中
- 缓存感知负载均衡（Cache-Aware Load Balancing）
  - 在多 GPU 或多节点部署的集群中，如果随机路由请求，可能会导致每个节点都无法形成高命中率的本地缓存。
    - 通过引入**缓存感知的路由调度器**（如 SGL Router），调度系统会分析请求的前缀，并尽量将具有相同前缀（如相同 System Prompt、同一场多轮对话）的请求分发到同一个 GPU 实例上，从而提升集群整体的 KV Cache 命中率。
- 语义/模糊 KV Cache 共享（Semantic KV Cache Reuse）
  - 目前标准的 Prefix Caching 要求 Token 序列**100% 精确匹配**（哪怕多一个空格或换行都会导致 Cache Miss）
    - 学术界与工业界（如 SemBlend 等技术）正在探索**语义级/模糊匹配**：通过向量检索或特定规则，识别出语义高度相似、仅有微小修改（如修改了个别无意义词、标点符号）的 Prompt，并通过特定的 Attention 算子或少量重计算技术复用大部分已有的 KV Cache，从而在 Prompt 存在细微扰动时依然能维持高缓存命中率。

## 参考资料

- https://medium.com/@rahularyan786/kv-cache-vs-radix-attention-vs-page-attention-941ff222be2e
- https://docs.vllm.ai/en/v0.6.0/automatic_prefix_caching/details.html
- https://docs.vllm.ai/en/stable/design/prefix_caching/
- https://medium.com/byte-sized-ai/prefix-caching-sglang-vs-vllm-token-level-radix-tree-vs-block-level-hashing-b99ece9977a1
- https://turion.ai/blog/vllm-vs-sglang-inference-comparison-2026/
- CPU Offloading：https://atlarge-research.com/pdfs/phan2025isp.pdf
- https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-25-sglang-72edc5043338f59db34d47e5b96ac870.pdf
- https://sgl-project-sglang-93.mintlify.app/concepts/radix-attention