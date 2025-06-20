### safe softmax

最原始的 `softmax` 的计算公式为:

$$
y=\mathrm{softmax}(x)
$$

$$
\begin{aligned}
y_i=\frac{e^{x_i}}{\sum_{j=1}^De^{x_j}}
\end{aligned}
$$

为了避免 $\sum_{j=1}^De^{x_j}$ 出现溢出的情况，可以做一个变形：

$$
y_{i}=\frac{e^{x_{i}-\max_{k=1}^{D}x_{k}}}{\sum_{j=1}^{D}e^{x_{j}-\max_{k=1}^{D}x_{k}}}
$$

其中,

$$\max_{k=1}^Dx_k$$

表示为整个序列的全局最大值。

##### 数学等价性证明

需证：

$$
\frac{e^{x_{i}}}{\sum_{j=1}^{n}e^{x_{j}}}=\frac{e^{z_{i}}}{\sum_{j=1}^{n}e^{z_{j}}}, z_i=x_i-m
$$

推导过程：

$$
\frac{e^{z_i}}{\sum_{j=1}^ne^{z_j}}=\frac{e^{x_i-m}}{\sum_{j=1}^ne^{x_j-m}}
$$

$$
=\frac{e^{x_i}\cdot e^{-m}}{\sum_{j=1}^n(e^{x_j}\cdot e^{-m})}=\frac{e^{x_i}\cdot e^{-m}}{e^{-m}\cdot\sum_{j=1}^ne^{x_j}}=\frac{e^{x_i}}{\sum_{j=1}^ne^{x_j}}
$$

得证。

##### 伪代码

```python
m = -inf
d = 0
for k in range(D):
 m = max(m, x[k])
for j in range(D):
 d += exp(x[j] - m)
for i in range(D):
 y[i] = exp(x[i] - m) / d
```

以上的做法也是通用的深度学习框架(tensorflow 等)所采用的方法。

如果想要再进一步进行优化，则需要了解 `online softmax`。