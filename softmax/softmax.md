### softmax

最原始的 `softmax` 的计算公式为:

$$ y=\operatorname{softmax}(x) $$

$$
\begin{aligned}
y_i=\frac{e^{x_i}}{\sum_{j=1}^De^{x_j}}
\end{aligned}
$$

伪代码:

```python
d = 0
for j in range(D):
 d += exp(x[j])
for i in range(D):
 y[i] = exp(x[i]) / d
```

以上就是最原始的 `softmax` 计算公式。