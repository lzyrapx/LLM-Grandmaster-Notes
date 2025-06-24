
基于 CUDA 的高性能矩阵乘法 (GEMM) 内核，利用 Tensor Core(MMA) 进行加速计算。包括：

- 使用 BF16 数据类型和 MMA 指令集加速计算
- 分块(tiling)策略优化内存访问
- 共享内存缓存数据减少全局内存访问
- 向量化内存加载(int4)提高带宽利用率
- 性能分析功能(时钟周期计数)

#### 普通 baseline 实现

```cpp
#include <cooperative_groups/memcpy_async.h>  // 异步内存拷贝
#include <cuda.h>
#include <mma.h>  // CUDA Matrix Multiply-Accumulate (MMA) API
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cstdlib>
#include <cuda/pipeline>  // CUDA流水线操作
#include <experimental/random>
#include <iostream>
#include <vector>

// GPU错误检查宏
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

// 内核错误检查宏
#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

const static bool CPU_DEBUG = true;  // 是否启用CPU调试验证
const static unsigned int Profiling_ROUNDS = 1000;  // 性能分析轮数

// 矩阵全局维度 (2048x2048x2048)
const static unsigned int M_GLOBAL = 2048;
const static unsigned int N_GLOBAL = 2048;
const static unsigned int K_GLOBAL = 2048;

// 线程块(Block)级别分块大小
const static unsigned int BLOCK_Tile_M = 64;
const static unsigned int BLOCK_Tile_N = 64;
const static unsigned int BLOCK_Tile_K = 64;

// 线程束(Warp)级别分块大小
const static unsigned int WARP_Tile_M = 32;
const static unsigned int WARP_Tile_N = 32;

// 共享内存偏移，避免bank conflict
const static unsigned int SKEW_BF16 = 0;

// Tensor Core MMA操作分块大小(m16n8k16)
const static unsigned int MMA_Tile_M = 16;
const static unsigned int MMA_Tile_N = 8;
const static unsigned int MMA_Tile_K = 16;

// 线程块(Block)的数量
const static unsigned int NUM_BlockTile_M = M_GLOBAL / BLOCK_Tile_M;
const static unsigned int NUM_BlockTile_N = N_GLOBAL / BLOCK_Tile_N;

// 每个Block中的Warp数量计算
const static unsigned int NUM_warp_m_per_block = BLOCK_Tile_M / WARP_Tile_M;
const static unsigned int NUM_warp_n_per_block = BLOCK_Tile_N / WARP_Tile_N;
const static unsigned int NUM_warp_block = NUM_warp_m_per_block * NUM_warp_n_per_block;

// 内存访问相关常量
const static unsigned int Block_TileK_Bytes = BLOCK_Tile_K * sizeof(__nv_bfloat16);  // sizeof(__nv_bfloat16) = 2
const static unsigned int Warp_TileK_Copy_Bytes = 32 * sizeof(int4);  // sizeof(int4) = 16
const static unsigned int Block_TileK_Copy_Lines_Per_Warp = Warp_TileK_Copy_Bytes / Block_TileK_Bytes;

// every Block_TileK_Copy_Line_LANEs threads copy one Block_TileK
const static unsigned int Block_TileK_Copy_Line_LANEs = (32 / Block_TileK_Copy_Lines_Per_Warp);

using namespace nvcuda;  // CUDA命名空间
using namespace cooperative_groups;  // 协作组API


// 调试函数：打印fragmentA矩阵
inline __device__ void check_fragementA(__nv_bfloat16* fragementA, int laneid,
                                        __nv_bfloat16* fragementA_global_vis) {
  unsigned int groupId = laneid >> 2;  // 每4个线程一组 (0-7)
  unsigned int threadID_in_group = laneid % 4;  // 组内线程ID (0-3)

  for (int i = 0; i < 8; i++) {
    int row = 0;
    int col = 0;
    // 计算8个元素在8x8矩阵中的位置
    if ((i >= 0 && i < 2) || (i >= 4 && i < 6)) {
      row = groupId;   // 上半部分行
    } else {
      row = groupId + 8;  // 下半部分行
    }
    // 计算列位置
    col = i < 4 ? ((threadID_in_group * 2) + (i & 0x1))
                : ((threadID_in_group * 2) + (i & 0x1) + 8);
    // 存储到全局内存
    fragementA_global_vis[row * 16 + col] = fragementA[i];
  }
  __syncthreads();
  if (laneid == 0) {
    // 主线程打印整个8x8矩阵
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
        printf("%-6.2f ", float(fragementA_global_vis[i * 16 + j]));
      }
      printf("\n");
    }
  }
}

/**
 * @brief 基于WMMA的矩阵乘法内核 (BF16输入, FP32输出)
 * 
 * @param matA 输入矩阵A (MxK, 行主序)
 * @param matB 输入矩阵B (NxK, 列主序)
 * @param matD 输出矩阵D (MxN)
 * @param d_gpu_clock GPU时钟周期计数器
 */
__global__ void mma_baseline(
    const __nv_bfloat16* __restrict__ matA,
    const __nv_bfloat16* __restrict__ matB,
    float* matD,
    long long int* d_gpu_clock /*, __nv_bfloat16* fragementA_global_vis*/) {
  
  // 协作组初始化
  auto this_grid = cooperative_groups::this_grid();
  auto this_block = cooperative_groups::this_thread_block();
  auto this_tile = tiled_partition<32>(this_block);  // 32 线程的 tile

  const unsigned int warpId = threadIdx.x / 32;   // Block 内的 warp ID (0-3)
  const unsigned int laneId = threadIdx.x % 32;   // Warp 内的 lane ID (0-31)
 
  // 动态共享内存声明 (用于矩阵分块)
  // 布局: [0-63][0-63] = matA分块, [64-127][0-63] = matB分块
  extern __shared__ __nv_bfloat16 buffer[][BLOCK_Tile_K + SKEW_BF16];

  // NUM_warp_n_per_block 是每个 block 中 warp 的数量
  // 计算当前warp在Block内的位置 (warp网格: 2x2)
  const unsigned int warpId_m = warpId / (NUM_warp_n_per_block);  // M 方向 warp 索引 (0-1)
  const unsigned int warpId_n = warpId % (NUM_warp_n_per_block);  // N 方向 warp 索引 (0-1)

  // 启动时钟周期计数
  long long int start_t = clock64();
  
  // 声明 WMMA 累加器寄存器 (每个 warp 计算 32x32 分块)
  // 这里的布局是: [2][4][4] -> 2(16x16分片) x 4(8x8分片) x 4(每个MMA的结果)
  float fragementD[(WARP_Tile_M / 16)][(WARP_Tile_N / 8)][4] = {};

  // 遍历所有 Block 分块
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    // 计算当前Block处理的矩阵分块坐标
    const unsigned int block_tile_m = block_pos / NUM_BlockTile_N;
    const unsigned int block_tile_n = block_pos % NUM_BlockTile_N;
    // 边界检查
    if (block_tile_m >= NUM_BlockTile_M) {  // boundary check
      break;
    }
    // K 方向分块循环 (用于累加)
    for (int blk_tile_k = 0; blk_tile_k < (K_GLOBAL / BLOCK_Tile_K); blk_tile_k++) {
      
      /********************** 矩阵A分块加载到共享内存 **********************/
      const unsigned int num_iters_ATile = BLOCK_Tile_M / (NUM_warp_block * Block_TileK_Copy_Lines_Per_Warp);
      const unsigned int num_lanes_workload_per_warp = BLOCK_Tile_M / NUM_warp_block;
      // 计算当前warp在共享内存中的起始位置
      unsigned int shmem_idx_warp = (num_lanes_workload_per_warp)*warpId;
      unsigned int shmem_idx_lane = shmem_idx_warp + (laneId / Block_TileK_Copy_Line_LANEs);
      // 计算全局内存中的位置
      unsigned int global_idx_lane_m = shmem_idx_lane + block_tile_m * BLOCK_Tile_M;
      unsigned int global_idx_lane_k =  laneId % Block_TileK_Copy_Line_LANEs;

       // 使用int4向量化加载 (每次加载16字节)
#pragma unroll
      for (int i = 0; i < num_iters_ATile; i++) {
        // 共享内存指针
        int4* shmem_ptr = (int4*)&buffer[shmem_idx_lane + i * Block_TileK_Copy_Lines_Per_Warp][0] + (laneId % Block_TileK_Copy_Line_LANEs);
        // 全局内存指针
        int4* global_ptr = (int4*)&matA[(global_idx_lane_m + i * Block_TileK_Copy_Lines_Per_Warp) * K_GLOBAL + blk_tile_k * BLOCK_Tile_K] + global_idx_lane_k;
        *shmem_ptr = *global_ptr;  // 执行向量化拷贝
      }

      // copy B, the difference is N dimension, K dimension should be same as MatA
      
      /********************** 矩阵B分块加载到共享内存 **********************/
      const unsigned int num_iters_BTile = BLOCK_Tile_N / (NUM_warp_block * Block_TileK_Copy_Lines_Per_Warp);
      const unsigned int num_lanes_workload_per_warp_b = BLOCK_Tile_N / NUM_warp_block;
      // 计算当前 warp 在共享内存中的起始位置 (B在共享内存的后半部分)
      unsigned int shmem_idx_warp_b = (num_lanes_workload_per_warp_b)*warpId;
      unsigned int shmem_idx_lane_n = shmem_idx_warp_b + (laneId / Block_TileK_Copy_Line_LANEs);
      unsigned int global_idx_lane_n = shmem_idx_lane_n + block_tile_n * BLOCK_Tile_N;
#pragma unroll
      for (int j = 0; j < num_iters_BTile; j++) {
        // 注意：B 矩阵存储在共享内存的 BLOCK_Tile_M 偏移处
        int4* shmem_ptr = (int4*)&buffer[shmem_idx_lane + j * Block_TileK_Copy_Lines_Per_Warp + BLOCK_Tile_M][0] + (laneId % Block_TileK_Copy_Line_LANEs);
        int4* global_ptr = (int4*)&matB[(global_idx_lane_n + j * Block_TileK_Copy_Lines_Per_Warp) * K_GLOBAL + blk_tile_k * BLOCK_Tile_K] + global_idx_lane_k;
        *shmem_ptr = *global_ptr;  // 执行向量化拷贝
      }

      // 同步确保所有数据加载完成
      __syncthreads();

      // Debug
      // if(block_pos == 1&& warpId==0 && laneId==0 && blk_tile_k == 0){
      //     printf("MatA in shared mem at blk_tile_k %d\n",blk_tile_k);
      //     for(int i=0;i<BLOCK_Tile_M;i++){
      //         for(int j=0;j<BLOCK_Tile_K;j++){
      //             printf("%-6.2f ",float(buffer[i][j]));
      //         }
      //         printf("\n");
      //     }
      // }

      // Debug
      // if(block_pos == 1 &&warpId==0 && laneId==0 && blk_tile_k==1){
      //     printf("MatB in shared mem at blk_tile_k %d\n",blk_tile_k);
      //     for(int i=0;i<BLOCK_Tile_M;i++){
      //         for(int j=0;j<BLOCK_Tile_K;j++){
      //             printf("%-6.2f ",float(buffer[i+BLOCK_Tile_M][j]));
      //         }
      //         printf("\n");
      //     }
      // }
      // break;

      /********************** WMMA计算核心 **********************/

      // 在 K 方向上进行小分块 MMA 计算
      for (int k_mma_step = 0; k_mma_step < (BLOCK_Tile_K / MMA_Tile_K); k_mma_step++) {
        // 声明寄存器存储矩阵分片
        __nv_bfloat16 fragementA[WARP_Tile_M / MMA_Tile_M][8] = {};  // 每个16x16分片
        __nv_bfloat16 fragementB[WARP_Tile_N / MMA_Tile_N][4] = {};  // 每个8x8分片

        // 遍历当前 warp 的 M 方向分片 (32/16=2)
#pragma unroll
        for (int mma_m = 0; mma_m < WARP_Tile_M / MMA_Tile_M; mma_m++) {
          // 从共享内存加载矩阵A分片 (使用 ldmatrix 指令)
          unsigned int* A = reinterpret_cast<unsigned int*>(&fragementA[mma_m][0]);
          // 计算共享内存中的矩阵A位置
          unsigned tile_matA_shared_ptr =
              static_cast<unsigned>(__cvta_generic_to_shared(
                &buffer[warpId_m * WARP_Tile_M + mma_m * MMA_Tile_M + laneId % 16][(laneId / 16) * 8 + k_mma_step * MMA_Tile_K]));
          
          // 内联PTX: 加载8x8 BF16矩阵到4个寄存器 (x4)
          asm volatile(
              "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3])
              : "r"(tile_matA_shared_ptr));

// if(warpId==0 &&  k_mma_step ==0 && mma_m==1){
//     if(laneId==0){
//         printf("fragementA at k_mma_step = %d, mma_m = %d, warp =
//         %d\n",k_mma_step,mma_m,warpId);
//     }
//     check_fragementA(fragementA[mma_m],laneId,fragementA_global_vis);
// }

          // 遍历当前 warp 的N方向分片 (32/8=4)
#pragma unroll
          for (int mma_n = 0; mma_n < WARP_Tile_N / MMA_Tile_N; mma_n++) {
            // 从共享内存加载矩阵B分片
            unsigned int* B = reinterpret_cast<unsigned int*>(&fragementB[mma_n][0]);
            // 计算共享内存中的矩阵B位置
            unsigned tile_matB_shared_ptr =
                static_cast<unsigned>(__cvta_generic_to_shared(
                    &buffer[warpId_n * WARP_Tile_N + mma_n * MMA_Tile_N + laneId % 8 + BLOCK_Tile_M][(laneId / 8) * 8 + k_mma_step * MMA_Tile_K]));
            
            // 内联PTX: 加载8x8 BF16矩阵到2个寄存器 (x2)
            asm volatile(
                "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];"
                : "=r"(B[0]), "=r"(B[1])
                : "r"(tile_matB_shared_ptr));

            // if(warpId==1 &&  k_mma_step ==0 && mma_m==0 && mma_n==0){
            //     if(laneId==0){
            //         printf("fragementB at k_mma_step = %d, mma_n = %d, warp =
            //         %d\n",k_mma_step,mma_n,warpId);
            //     }
            //     for(int i=0;i<4;i++){
            //         printf("laneid = %d, fragementB[%d] =
            //         %-6.2f\n",laneId,i,float(fragementB[mma_n][i]));
            //     }
            // }

            // float *C = reinterpret_cast<float *>(fragementD[mma_m][mma_n]);
            // 获取当前累加器指针
            float* D = reinterpret_cast<float*>(fragementD[mma_m][mma_n]);

            // 内联PTX: 执行m16n8k16 MMA操作 (BF16输入, FP32累加)
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                "{%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]),
                  "r"(B[1]), "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));

            // if(warpId==0 &&  k_mma_step == 1 && mma_m==0 && mma_n==0){
            //     if(laneId==0){
            //         printf("fragementD at k_mma_step = %d, mma_m = %d, mma_n
            //         = %d, warp = %d\n",k_mma_step,mma_m,mma_n,warpId);
            //     }
            //     for(int i=0;i<4;i++){
            //         printf("laneid = %d, fragementD[%d] =
            //         %-6.2f\n",laneId,i,float(fragementD[mma_m][mma_n][i]));
            //     }
            // }
          }
        }
      }
      // 确保所有 warp 完成计算
      __syncthreads();
    }

    // #ifdef DEBUG_KERNEL
    // if(warpId==0 ){
    //     if(laneId==0){
    //         printf("fragementD at mma_m = %d, mma_n = %d, warp =
    //         %d\n",0,0,warpId);
    //     }
    //     for(int i=0;i<4;i++){
    //         printf("laneid = %d, fragementD[%d] =
    //         %-6.2f\n",laneId,i,float(fragementD[0][0][i]));
    //     }
    // }
    // #endif

    // synchronize the block so all data is ready
    this_block.sync();

    long long int end_t = clock64();

    // if(warpId==0 && laneId==0){
    //     printf("num cycles of mma baseline %lld at blockID = %d\n",
    //     end_t-start_t,block_pos);
    // }

    // 记录内核执行时间
    if (laneId == 0) {
      d_gpu_clock[block_pos * NUM_warp_block + warpId] = end_t - start_t;
    }

    // note below implementation is used for verifying GPU resluts, streaming
    // the result back to global mem can be optimized this lazy implemention
    // gives poor performance since we did not consider memory coalesces

    /********************** 结果写回全局内存 **********************/
    
    // 将寄存器中的结果写回全局内存 (此部分可优化)
    
    int group_id = laneId >> 2;  // 每4个线程一组 (0-7)
    int threadID_in_group = laneId % 4;  // 组内线程ID (0-3)

    // 遍历warp计算的所有分片
    for (int i = 0; i < WARP_Tile_M / 16; i++) {  // M 方向分片 (2)
      for (int j = 0; j < WARP_Tile_N / 8; j++) {  // N 方向分片 (4)
        for (int k = 0; k < 4; k++) {    // 每个分片的 4 个结果
          // 计算结果在分片内的位置
          int row = k < 2 ? group_id : (group_id + 8);
          int col = (threadID_in_group * 2) + (k & 0x1);
          
          // 计算在 warp 分块中的位置
          row = row + i * 16;
          col = col + j * 8;
          
          // 计算在 Block 分块中的位置
          row = row + warpId_m * WARP_Tile_M;
          col = col + warpId_n * WARP_Tile_N;

          // 计算全局内存位置
          row = BLOCK_Tile_M * block_tile_m + row;
          col = BLOCK_Tile_N * block_tile_n + col;
          
          // if(warpId == 0 && laneId == 0){
          //     printf("store fragementD[%d][%d][%d] = %-6.2f to matD[%d][%d]\n",i,j,k,fragementD[i][j][k],row,col);
          // }
          // 写回全局内存
          matD[row * N_GLOBAL + col] = fragementD[i][j][k];
        }
      }
    }
    // 确保所有线程完成写回
    __syncthreads();
  }
}

/**
 * @brief 初始化主机矩阵
 * 
 * @param a 矩阵A (MxK, 行主序)
 * @param b 矩阵B (NxK, 列主序)
 */
__host__ void init_host_matrices(__nv_bfloat16* a, __nv_bfloat16* b) {
  // 初始化矩阵A (行主序)
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      a[i * K_GLOBAL + j] = (__nv_bfloat16)(float)(rand() % 4);
    }
  }
  // 初始化矩阵B (列主序)
  for (int i = 0; i < N_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      b[i * K_GLOBAL + j] = (__nv_bfloat16)(float)((rand()) % 4);
    }
  }
}

/**
 * @brief CPU参考实现 (矩阵乘法)
 * 
 * @param matrixA 输入矩阵A
 * @param matrixB 输入矩阵B
 * @param gemmM M维度
 * @param gemmN N维度
 * @param gemmK K维度
 * @param MatrixD 输出矩阵
 */
__host__ void compute_cpu(__nv_bfloat16* matrixA, __nv_bfloat16* matrixB,
                          int gemmM, int gemmN, int gemmK, float* MatrixD) {
  for (int row = 0; row < gemmM; row++) {
    for (int col = 0; col < gemmN; col++) {
      float tmp = 0.0;
      for (int k = 0; k < gemmK; k++) {
        tmp += float(matrixA[row * gemmK + k]) * float(matrixB[col * gemmK + k]);
        // tmp += float(matrixA[row * gemmK/2 + k].y) * float(matrixB[col*gemmK/2 + k].y);
      }
      MatrixD[row * gemmN + col] = tmp;
    }
  }
}

/**
 * @brief 打印矩阵 (调试用)
 */
__host__ void printMatrix(float* matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%-6.2f ", float(matrix[i * cols + j]));
    }
    printf("\n");
  }
}


__host__ void printMatrix(__nv_bfloat16* matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%-6.2f ", float(matrix[i * cols + j]));
    }
    printf("\n");
  }
}

/**
 * @brief 比较两个矩阵是否相等
 * 
 * @return true 矩阵相等
 * @return false 矩阵不相等
 */
__host__ bool compare_two_matrix(float* matrixA, float* matrixB, int rows,
                                 int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (matrixA[i * cols + j] != matrixB[i * cols + j]) {
        printf("matrixA[%d][%d] (%-6.2f) != matrixB[%d][%d] (%-6.2f)\n", i, j,
               matrixA[i * cols + j], i, j, matrixB[i * cols + j]);
        return false;
      }
    }
    // printf("\n");
  }
  printf("Two input matrices are same\n");
  return true;
}

int main() {
  // 分配主机内存
  __nv_bfloat16* matA_cpu = new __nv_bfloat16[M_GLOBAL * K_GLOBAL];

  __nv_bfloat16* matB_cpu = new __nv_bfloat16[N_GLOBAL * K_GLOBAL];

  float* cpu_result = new float[M_GLOBAL * N_GLOBAL];

  // 初始化主机矩阵
  init_host_matrices(matA_cpu, matB_cpu);

  // std::cout<<"print MatA"<<std::endl;
  // printMatrix(matA_cpu,M_GLOBAL,K_GLOBAL);

  // std::cout<<"print MatB"<<std::endl;
  // printMatrix(matB_cpu,N_GLOBAL,K_GLOBAL);

  // 计算内核执行参数
  int num_blocks = (M_GLOBAL / BLOCK_Tile_M) * (N_GLOBAL / BLOCK_Tile_N);  // Block 数量
  int num_threads_per_block = NUM_warp_block * 32;  // 每个 Block 的线程数 (4 warps * 32 = 128)

  // 共享内存大小
  int size_shmem_per_block_bytes = (BLOCK_Tile_M * (BLOCK_Tile_K + SKEW_BF16) +
                                    BLOCK_Tile_N * (BLOCK_Tile_K + SKEW_BF16)) *
                                   sizeof(__nv_bfloat16);

  // 分配设备内存
  __nv_bfloat16* d_matA = nullptr;

  __nv_bfloat16* d_matB = nullptr;
  float* d_matD = nullptr;
  float* h_matD = new float[M_GLOBAL * N_GLOBAL];  // 从设备拷贝的结果

  long long int* d_gpu_clock = nullptr;
  long long int* h_gpu_clock = new long long int[num_blocks * NUM_warp_block];

  __nv_bfloat16* d_fragementA_vis = nullptr;

  // 分配设备内存
  gpuErrchk(cudaMalloc(&d_gpu_clock, num_blocks * NUM_warp_block * sizeof(long long int)));
  gpuErrchk(cudaMalloc(&d_matA, M_GLOBAL * K_GLOBAL * sizeof(__nv_bfloat16)));
  gpuErrchk(cudaMalloc(&d_matB, N_GLOBAL * K_GLOBAL * sizeof(__nv_bfloat16)));
  gpuErrchk(cudaMalloc(&d_matD, M_GLOBAL * N_GLOBAL * sizeof(float)));

  gpuErrchk(cudaMalloc(&d_fragementA_vis, MMA_Tile_M * MMA_Tile_K * sizeof(__nv_bfloat16)));

  // 拷贝数据到设备
  gpuErrchk(cudaMemcpy(d_matA, matA_cpu,
                       M_GLOBAL * K_GLOBAL * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_matB, matB_cpu,
                       N_GLOBAL * K_GLOBAL * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice));

  // 创建CUDA事件用于计时
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 性能分析循环
  long long int baseline_cycles_total = 0;
  int NUM_PROFILES = Profiling_ROUNDS;
  for (int iter = 0; iter < NUM_PROFILES; ++iter) {
    // float milliseconds = 0;
    gpuErrchk(cudaMemset(d_gpu_clock, 0, num_blocks * sizeof(long long int)));
    cudaEventRecord(start);

    // 启动内核
    mma_baseline<<<num_blocks, num_threads_per_block,
                   size_shmem_per_block_bytes>>>(
        d_matA, d_matB, d_matD, d_gpu_clock /*,d_fragementA_vis*/);

    // 拷贝计时结果
    gpuErrchk(cudaMemcpy(h_gpu_clock, d_gpu_clock,
                         num_blocks * NUM_warp_block * sizeof(long long int),
                         cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    
    // cudaEventElapsedTime(&milliseconds, start, stop);

    // 统计最大时钟周期
    baseline_cycles_total += *std::max_element(h_gpu_clock, h_gpu_clock + num_blocks * NUM_warp_block);
  }

  // 计算平均时钟周期
  long long int baseline_cycles = (baseline_cycles_total) / (NUM_PROFILES);
  printf("num of cycles mma baseline: %lld\n", baseline_cycles);

  gpuErrchk(cudaPeekAtLastError());
  
  // 拷贝结果回主机
  gpuErrchk(cudaMemcpy(h_matD, d_matD, M_GLOBAL * N_GLOBAL * sizeof(float),
                       cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();

  // if (CPU_DEBUG == true) {
  //   printf("Verifying GPU result against CPU reference...\n");
  //   compute_cpu(matA_cpu, matB_cpu, M_GLOBAL, N_GLOBAL, K_GLOBAL, cpu_result);
  //   printf("compute on cpu done\n");
  //   bool check = compare_two_matrix(cpu_result, h_matD, M_GLOBAL, N_GLOBAL);
  //   if(check == false) {
  //     printf("check failed\n");
  //     // std::cout<<"print GPU result"<<std::endl;
  //     // printMatrix(h_matD,M_GLOBAL,N_GLOBAL);
  //     // std::cout<<"print CPU reference"<<std::endl;
  //     // printMatrix(cpu_result,M_GLOBAL,N_GLOBAL);
  //   } else {
  //     printf("check passed\n");
  //   }
  // }

  // 释放资源
  delete[] matA_cpu;
  delete[] matB_cpu;
  delete[] cpu_result;
  delete[] h_matD;
  delete[] h_gpu_clock;
  
  cudaFree(d_matA);
  cudaFree(d_matB);
  cudaFree(d_matD);
  cudaFree(d_gpu_clock);
}

/*
export CUDA_PATH=/usr/local/cuda-12.4
export PATH=$CUDA_PATH/bin:$PATH
export CUDACXX=$CUDA_PATH/bin/nvcc
export TargetSM=80

A100:
num of cycles mma baseline: 909533
H100:
num of cycles mma baseline: 597198
*/
```
