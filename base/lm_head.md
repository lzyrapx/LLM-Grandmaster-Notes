## lm_head

语言模型头，就是一个矩阵，把结果映射回 `vocab`，然后判断哪一个词的概率最大。

e.g.
Gemma 模型的 `LM Head`（语言模型头） 是一个线性层（`nn.Linear`），其作用是将模型最后一层的隐藏状态（`hidden states`）映射到词汇表空间，用于生成下一个 `token` 的概率分布。

## Gemma 的 LM Head

以下是 gemma 的模型结构，可以看到 `lm head` 在模型结构的最后一层。

```python
GemmaForCausalLM(
  (model): GemmaModel(
    (embed_tokens): Embedding(256000, 2048, padding_idx=0)
    (layers): ModuleList(
      (0-17): 18 x GemmaDecoderLayer(
        (self_attn): GemmaSdpaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=256, bias=False)
          (v_proj): Linear(in_features=2048, out_features=256, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): GemmaRotaryEmbedding()
        )
        (mlp): GemmaMLP(
          (gate_proj): Linear(in_features=2048, out_features=16384, bias=False)
          (up_proj): Linear(in_features=2048, out_features=16384, bias=False)
          (down_proj): Linear(in_features=16384, out_features=2048, bias=False)
          (act_fn): PytorchGELUTanh()
        )
        (input_layernorm): GemmaRMSNorm()
        (post_attention_layernorm): GemmaRMSNorm()
      )
    )
    (norm): GemmaRMSNorm()
  )
  (lm_head): Linear(in_features=2048, out_features=256000, bias=False)
)
```

### LM Head 的结构

输入维度：等于模型的隐藏层维度（`hidden_size`），例如：

- Gemma-2B：隐藏层维度为 `2048`
- Gemma-7B：隐藏层维度为 `4096`

输出维度：等于词汇表大小（`vocab_size`），Gemma 的词汇表大小为 `256,000`。

参数数量：`hidden_size × vocab_size`，例如：

Gemma-7B：`4096 × 256,000 = 1,048,576,000`（约 1B 参数）

### LM Head 的参数量占比

在 Gemma-7B 中，LM Head 的参数量约为 1B，占模型总参数（7B）的 14% 左右。

在 Gemma-2B 中，LM Head 的参数量约为 `2048 × 256,000 = 524,288,000`（约 0.5B），占总参数的 25% 左右。

可以看到参数量也不少，必要时可以根据场景对 `lm head` 进行裁剪。

### 实现细节

Gemma 的 `LM Head `没有额外的归一化或激活函数，直接通过线性投影生成 `logits`。

在代码（如 Hugging Face 实现）中，通常定义为：

```python
self.lm_head = nn.Linear(hidden_size, vocab_size)
```

### 为什么 LM Head 参数量大？

由于词汇表规模大（`256k`），即使隐藏层维度较高（如 `4096`），线性层的参数量也会显著增加。这是大语言模型的典型设计。