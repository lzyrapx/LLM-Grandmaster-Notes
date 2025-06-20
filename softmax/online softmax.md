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

其中 $d_j$ 代表前 $j$ 个元素的指数和。该式可以通过数学归纳法证明:

$$
d_j=\sum_{k=1}^je^{x_k-max_j}
$$

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


