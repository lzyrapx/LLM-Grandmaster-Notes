# Flash Attention

## Self-Attention

$$
S = QK^T,(N \times N)
$$
$$
P = softmax(S), (N \times N)
$$
$$
O = PV, (N \times d)
$$

### Safe Softmax

$$
safe\ softmax = \frac{e^{x_i -m}}{\sum_{j=1}^{N}e^{x_j-m}}
$$

计算 safe-softmax 的时候需要对 `[1, N]` 重复三次，需要访问 `Q` 和 `K` 三次， 并实时重新计算 `x`，这样会很低效，所以需要将计算进行合并。

现在定义 

$$
d_{i}^{'} = \sum_{j=1}^{i}e^{x_j - m_i}
$$

当 $i = N$ 的时候，有：

$$
d_N = d_{N}^{'} = \sum_{j=1}^{i}e^{x_j - m_N}
$$

$d_{i}^{'}$ 和 $d_{i-1}^{'}$ 有如下递归关系：

$$
d_{i}^{'} = \sum_{j=1}^{i}e^{x_j - m_i} \\
= (\sum_{j=1}^{i-1}e^{x_j - m_i}) + e^{x_i - m_i} \\
= (\sum_{j=1}^{i-1}e^{x_j - m_{i-1}})e^{m_{i-1} - m_i} + e^{x_i - m_i} \\
= d_{i-1}^{'}e^{m_{i-1} - m_i} + e^{x_i - m_i}
$$

递归关系只依赖 $m_{i-1}$ 和 $m_i$，于是可以把 $d_i^{'}$ 和 $m_i$ 放在同一个循环中。

## flash attention 1

online softmax 最多只有一个 `2-pass` 的算法，不存在 `1-pass` 算法，但是 `attention` 可以有 `1-pass` 算法。基于上述的 online softmax 可以得到一个 `1-pass`的 attention 算法。

重点在第二个循环：

$$
a_i = \frac{e^{x_i - m_N}}{d_N^{'}} \\
o_i = o_{i-1} + a_i V[i,:]
$$

推导 `1-pass` 版本的 flash attention:

$$
o_i^{'} = (\sum_{j=1}^i (\frac{e^{x_j - m_i}}{d_i^{'}})V[j,:]) \\
$$

当 $i= N$ 时有：

$$
o_N^{'} = o_N = (\sum_{j=1}^i (\frac{e^{x_j - m_N}}{d_N^{'}})V[j,:]) \\
$$

推导 $o_{i}{'}$ 和 $o_{i-1}{'}$ 之间的关系：

$$
o_i^{'} = (\sum_{j=1}^i (\frac{e^{x_j - m_i}}{d_i^{'}})V[j,:])
$$
$$
= (\sum_{j=1}^{i-1} (\frac{e^{x_j - m_i}}{d_i^{'}})V[j,:]) + (\frac{e^{x_i - m_i}}{d_i^{'}})V[i,:]
$$
$$
= (\sum_{j=1}^{i-1} (\frac{e^{x_j - m_{i-1}}}{d_i^{'}})\frac{e^{x_j-m_i}}{e^{x_j-m_{i-1}}}\frac{d_{i-1}^{'}}{d_{i}^{'}}V[j,:]) + (\frac{e^{x_i - m_i}}{d_i^{'}})V[i,:]
$$
$$
= o_{i-1}^{'}\frac{d_{i-1}^{'}e^{m_{i-1} - m_i}}{d_i^{'}} + (\frac{e^{x_i - m_i}}{d_i^{'}})V[i,:]
$$

可以看到 $o_i^{'}$ 和 $o_{i-1}^{'}$ 递归关系不依赖 $m_n$，因此可以将第二个循环完全合并到第一个循环中去。

## flash attention 2

主要做了工程上的优化：
- 减少大量非矩阵乘的冗余计算，增加 Tensor Core 的计算比例。
- forward pass/backward pass 均增加 seq_len 维度的并行，forward pass 交替 Q，K，V 循环顺序。
- 更好的 Warp Partitioning 策略，避免 Split-K。

## flash attetion 3

主要做了针对 hopper 架构的优化：
- 利用 Tensor Core、TMA 和 Warp-Specialization 等特性
