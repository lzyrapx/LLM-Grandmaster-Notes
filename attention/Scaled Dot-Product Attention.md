## Scaled Dot-Product Attention (SDPA, 单头注意力)

在理解多头之前，先回顾一下单头注意力 `SDPA` 的计算 `Attention(Q, K, V)`:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^{T}}{\sqrt{d_k}})V
$$

- 输入表示： 假设有一个输入序列，包含 `n` 个元素（例如单词），每个元素被表示为一个 `d_model` 维的向量。我们将整个序列堆叠成一个矩阵 `X` (形状 `n x d_model`)
- 线性投影 (`Query`, `Key`, `Value`)：
  - 使用三个不同的可学习权重矩阵 `W^Q`, `W^K`, `W^V` (每个形状 `d_model x d_k` 或 `d_model x d_v`，通常 `d_k = d_v`)。
  - `Query` 向量： `Q = X * W^Q` (形状 `n x d_k`) - 代表我们要“询问”序列中哪些部分信息。
  - `Key` 向量： `K = X * W^K` (形状 `n x d_k`) - 代表序列中每个元素可以用来被“查询”的标识。
  - `Value` 向量： `V = X * W^V` (形状 `n x d_v`) - 代表序列中每个元素实际包含的信息。
- 计算注意力分数（`Dot-Product`）： 计算 `Query` 向量和所有 `Key` 向量之间的相似度（点积）。分数矩阵 `S = Q * K^T` (形状 `n x n`)。
- 缩放（`Scaling`）：将注意力分数除以 `sqrt(d_k)`。这一步非常重要，因为点积在 `d_k` 较大时其值会非常大，导致 `softmax` 梯度极小，防止梯度爆炸或消失。
- `Softmax` 归一化： 对缩放后的分数矩阵的每一行应用 `softmax` 函数。这使得每一行的分数总和为 `1`，并产生 注意力权重矩阵 `A` (形状 `n x n`)。`A[i, j]` 表示第 `i` 个元素在生成其输出表示时，应该给予第 `j` 个元素多少“注意力”。
- 加权求和： 将 `Value` 矩阵 `V` 用注意力权重 `A` 加权求和，得到输出矩阵 `Z` (形状 `n x d_v`)： `Z = A * V`。
  - `Z[i] = sum_j(A[i, j] * V[j])` - 第 `i` 个元素的输出是其自身 `Value` 和所有其他元素 `Value` 的加权和，权重由它与所有其他元素的相关性决定。

`SDPA` 的意义： 它允许序列中的每个位置在生成自己的新表示时，能够“关注”序列中所有其他位置（包括自身）的信息，并根据相关性动态地聚合这些信息。

## SDPA 和 Self-Attention 区别

|特性|Scaled Dot-Product Attention|Self-Attention|
|:---:|:---:|:---:|
|层级|底层计算机制 (具体的公式、算法)|高层应用模式/概念 (如何配置和使用注意力)|
|定义核心|给定 `Q`, `K`, `V`，如何计算输出 `Z`|`Q`, `K`, `V` 必须来自同一个输入序列 `X`|
|关注点|计算过程本身 (点积、缩放、`softmax`、加权求和)|输入来源 (`Q`, `K`, `V` 同源) 和 目的 (学习序列内部关系)|
|是否依赖输入来源|否。公式不关心 `Q`, `K`, `V` 的来源。|是。核心定义就是 `Q`, `K`, `V` 同源。|
|目的|提供一种计算注意力加权重输出的通用方法|学习同一个序列元素之间的内部依赖关系|
|关系|`Self-Attention` 使用 `Scaled Dot-Product` 来计算|`Scaled Dot-Product` 是 `Self-Attention` 常用的计算引擎|

## Pytorch 实现

```python
import torch
import torch.nn.functional as F

def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor,
          M: int, N: int, d: int):
    # Q, K, V, output are tensors on the gpu
    
    # Reshape inputs from flattened arrays to 2D matrices
    Q_2d = Q.view(M, d)
    K_2d = K.view(N, d)
    V_2d = V.view(N, d)
    
    # Compute Q*K^T divided by sqrt(d)
    scale_factor = torch.sqrt(torch.tensor(d, dtype=torch.float32, device=Q.device))
    scores = torch.matmul(Q_2d, K_2d.t()) / scale_factor
    
    # Apply softmax row-wise (along dim=1)
    # PyTorch's softmax handles numerical stability automatically
    attention_weights = F.softmax(scores, dim=1)
    
    # Compute the final output: attention_weights * V
    result = torch.matmul(attention_weights, V_2d)
    
    # Copy to output tensor
    output.copy_(result.view(-1))
def main():
    # Set manual seed for reproducibility
    torch.manual_seed(42)
    
    # Configuration: 2 queries, 3 key-value pairs, 4-dimensional embeddings
    M, N, d = 2, 3, 4
    
    # Create test data (flattened tensors)
    Q = torch.randn(M * d, device='cuda')
    K = torch.randn(N * d, device='cuda')
    V = torch.randn(N * d, device='cuda')
    output = torch.zeros(M * d, device='cuda')
    
    # Print inputs
    print("Input Q:", Q.view(M, d).cpu())
    print("Input K:", K.view(N, d).cpu())
    print("Input V:", V.view(N, d).cpu())
    
    # Compute attention using custom implementation
    solve(Q, K, V, output, M, N, d)
    custom_result = output.view(M, d).cpu()
    
    # Compute reference using PyTorch's built-in attention
    Q_2d = Q.view(M, d)
    K_2d = K.view(N, d)
    V_2d = V.view(N, d)
    attn = F.scaled_dot_product_attention(
        Q_2d.unsqueeze(0), K_2d.unsqueeze(0), V_2d.unsqueeze(0)
    ).squeeze(0)
    
    # Compare results
    print("\nCustom implementation result:")
    print(custom_result)
    
    print("\nPyTorch built-in attention result:")
    print(attn.cpu())
    
    # Check numerical equivalence
    assert torch.allclose(custom_result, attn.cpu(), atol=1e-6), "Results mismatch!"
    print("\nTest passed: Results match!")

if __name__ == "__main__":
    main()
```

## CUDA 实现

```C++
#include <cuda_runtime.h>
#include <math.h>

// 计算 M×N 的矩阵。每个线程处理 Q 的第 i 行和 K 的第 j 行的点积。
__global__ void qkt_kernel(const float* Q, const float* K, float* S, int M, int N, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < d; ++k) {
        sum += Q[i * d + k] * K[j * d + k]; // Q[i] · K[j]
    }
    sum /= sqrtf(d);
    S[i * N + j] = sum;
}

// 处理每行的 N 个元素，保证执行 row wise 的 softmax
__global__ void softmax_kernel(const float* S, float* P, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // 计算行最大值
    float max_val = -INFINITY;
    for (int j = tid; j < N; j += num_threads) {
        max_val = fmaxf(max_val, S[row * N + j]);
    }

    // 归约求最大值
    __shared__ float shared_max[256];
    shared_max[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    float row_max = shared_max[0];

    // 计算 exp 和总和
    float sum_exp = 0.0f;
    for (int j = tid; j < N; j += num_threads) {
        float exp_val = expf(S[row * N + j] - row_max);
        P[row * N + j] = exp_val;
        sum_exp += exp_val;
    }

    // 归约求和
    __shared__ float shared_sum[256];
    shared_sum[tid] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float row_sum = shared_sum[0];

    // 归一化
    for (int j = tid; j < N; j += num_threads) {
        P[row * N + j] /= row_sum;
    }
}

// 遍历 N 次循环，确保 P 与 V 的正确矩阵乘法
__global__ void pv_kernel(const float* P, const float* V, float* output, int M, int N, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || k >= d) return;

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum += P[i * N + j] * V[j * d + k];
    }
    output[i * d + k] = sum;
}

void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float *S, *P;
    cudaMalloc(&S, M * N * sizeof(float));
    cudaMalloc(&P, M * N * sizeof(float));

    // 计算 QK^T / sqrt(d)
    dim3 block_qkt(16, 16);
    dim3 grid_qkt((M + 15) / 16, (N + 15) / 16);
    qkt_kernel<<<grid_qkt, block_qkt>>>(Q, K, S, M, N, d);
    cudaDeviceSynchronize();

    // 应用 softmax
    softmax_kernel<<<M, 256>>>(S, P, M, N);
    cudaDeviceSynchronize();

    // 计算 PV
    dim3 block_pv(16, 16);
    dim3 grid_pv((M + 15) / 16, (d + 15) / 16);
    pv_kernel<<<grid_pv, block_pv>>>(P, V, output, M, N, d);
    cudaDeviceSynchronize();

    cudaFree(S);
    cudaFree(P);
}
```