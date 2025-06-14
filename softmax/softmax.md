### softmax

最原始的 `softmax` 的计算公式为:

$$
y=\mathrm{softmax}(x)
$$

$$
\begin{aligned}
y_i=\frac{e^{x_i}}{\sum_{j=1}^De^{x_j}}
\end{aligned}
$$

$D$ 代表输入向量 $x$ 的维度（长度），也就是类别或输出节点的总个数。
比如，假设有一个包含 $D$ 个数值的向量 $x = [x_1, x_2, ..., x_i, ..., x_D]$。这些数值通常是神经网络最后一层（输出层）的原始输出值，称为 `logits`。

伪代码:

```python
d = 0
for j in range(D):
 d += exp(x[j])
for i in range(D):
 y[i] = exp(x[i]) / d
```

以上就是最原始的 `softmax` 计算公式。