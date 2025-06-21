## online softmax

### paper

[Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867)

### 介绍


safe softmax 伪代码：

```python
max = -inf
d = 0
for k in range(D):
 max = max(max, x[k])
for j in range(D):
 d += exp(x[j] - max)
for i in range(D):
 y[i] = exp(x[i] - max) / d
```

可以看到，为了计算一个 `softmax`，对输入遍历了 3 遍，对每一个元素，总共有 4 次访存（对 $x$ 有 3 次读取，对 $y$ 有 1 次写入）。

1. 第一遍计算出最大值
2. 第二遍计算出指数和
3. 第三遍计算输出

鉴于上述效率低下的问题，NVIDIA 在 2018 年提出了一个新的高效计算方法：`Online Softmax`，其主要是将上述第一次和第二次的遍历合二为一，只需要遍历一次就可以拿到最大值和指数和。具体做法如下：

$$
d_j=d_{j-1}\times e^{max_{j-1}-max_j}+e^{x_j-max_j}
$$

其中 $d_j$ 代表前 $j$ 个元素的指数和。该式可以通过数学归纳法证明以下公式:

$$
d_j=\sum_{k=1}^je^{x_k-max_j}
$$

**证明**：

定义：

$$\max_{j}=\max\left\{x_{1}, x_{2},\ldots, x_{j}\right\}$，即前  $j$  个元素的最大值。

**要证明**：

$$d_{j}=\sum_{k=1}^{j}e^{x_{k}-\text {max}_{j}}=d_{j-1}\times e^{\max_{j-1}-\max_{j}}+e^{x_{j}-\max_{j}}$$

递推公式：对于 $j \geq 2$ ，有 $d_{j}=d_{j-1} \times e^{\max_{j-1}-\max_{j}}+e^{x_{j}-\max_{j}}$，且基础情况 $j=1$ 时，$d_{1}=e^{x_{1}-\text{max}_{1}}=e^{x_{1}-x_{1}}=e^{0}=1$ （因为 $\max_{1}=x_{1}$）。

1. 基础情况 ($j = 1$)
- 当 $j=1$ ：
    - $\max_{1}=x_{1}$ （只有一个元素）。
    - 目标： $d_{1}=\sum_{k=1}^{1} e^{x_{k}-\text{max}_{1}}=e^{x_{1}-x_{1}}=e^{0}=1$ 。
    - 递推公式未定义（因为 $j-1=0$ 不存在），但根据定义直接计算 $d_{1}=1$，符合目标。基础情况成立。
2. 归纳假设
- 假设对于 $j-1$ (其中 $j\geq2$ )，公式成立，即：

$$d_{j-1}=\sum_{k=1}^{j-1}e^{x_k-\max_{j-1}}\text{。}$$

- 需要证明：对于 $j$ ,有 $d_j=\sum_{k=1}^je^{x_k-\max_{j}}$, 且使用递推公式 $d_j=d_{j-1}\times e^{\max_{j-1}-\max_{j}}+e^{x_j-\max_{j}}$ 计算后等价于目标。

注意到 $\max_{j}=\max\{\max_{j-1},x_j\}$，因此有两种可能情况：要么 $\max_{j}=\max_{j-1}$ (当$x_j\leq\max_{j-1})$，要么 $\max_j=x_j$ (当 $x_j>\max_{j-1}$ )。需要分情况证明。

**情况 1**：

当 $\max_{j}=\max_{j-1}$，(即 $x_j\leq\max_{j-1}$)

此时，$\max_{j-1}-\max_{j}=0$，所以 $e^{\max_{j-1}-\max_{j}}=e^0=1$。
递推公式简化为：

$$d_j=d_{j-1}\times1+e^{x_j-\max_{j}}=d_{j-1}+e^{x_j-\max_{j}}$$

由归纳假设，$d_{j-1}=\sum_{k=1}^{j-1}e^{x_k-\max_{j-1}}$。

由于 $\max_{j}=\max_{j-1}$，有 $\sum_{k=1}^{j-1}e^{x_k-\max_{j-1}}=\sum_{k=1}^{j-1}e^{x_k-\max_{j}}$

代入得：

$$d_j=\sum_{k=1}^{j-1}e^{x_k-\max_{j}}+e^{x_j-\max_{j}}=\sum_{k=1}^je^{x_k-\max_{j}}$$。

已证明。

情况 2: $\max_{j}=x_j$，即 $x_j>\max_{j-1}$

此时，$\max_{j} = x_{j}$，且 $\max_{j-1}-\max_{j}=\max_{j-1}-x_{j}$（注意：$\max_{j-1} - x_{j} < 0$，但指数计算仍有效）。

递推公式为：

$d_{j} = d_{j-1} \times e^{\max_{j-1}-\max_{j}}+e^{x_{j}-\max_{j}} = d_{j-1} \times e^{\max_{j-1} - x_{j}} + e^{x_{j} - x_{j}}=d_{j-1}\times e^{\max_{j-1} - x_{j}} + 1$

伪代码：

```python
max = -inf * ones(D)
d = zeros(D)

max[0] = x[0]
d[0] = x[0]
for j in range(1, D):
    max[j] = max(max[j - 1], x[j])
    d[j] = d[j - 1] * exp(m[j - 1] - max[j]) + exp(x[j] - max[j])

for i in range(D):
    y[i] = exp(x[i] - max[-1]) / d[-1]
```

可以看出来 `online softmax` 对每个元素只需要 3 次访存（对 $x$ 有 2 次，对 $y$ 有 1 次）。总共只需要遍历两次即可完成 `softmax` 的求解。

回顾一下 `softmax` 的过程：

1. 先对整个序列过一遍，拿到整个序列的全局最大值
2. 再对整个序列过一遍，拿到整个序列的指数和
3. 对序列过第三遍，进行归一化

而 `Online Softmax` 的过程：

1. 先对整个序列过一遍，同时拿到整个序列的全局最大值和指数和
2. 再对整个序列过一遍，进行归一化
