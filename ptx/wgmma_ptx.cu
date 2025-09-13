#include <cstdlib>
#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
using namespace cute;

__host__ __device__ void
print1(half_t v)
{
    printf("%*.0f ", 2, float(v));
}

__host__ __device__ void
print1(float v)
{
    printf("%*.0f ", 2, float(v));
}

template <class TA, class TB, class TC>
void print_matrix(TA *A, TB *B, TC *C, int m, int n, int k, int lda, int ldb)
{
    printf("m = %d, n = %d, k = %d\n", m, n, k);
    printf("The physical order of A:\n");
    for (int i = 0; i < m * k; ++i)
    {
        print1(A[i]);
    }
    printf("\n\n");

    printf("The physical order of B:\n");
    for (int i = 0; i < n * k; ++i)
    {
        print1(B[i]);
    }
    printf("\n\n");

    printf("The logical shape of A:\n");
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if (lda == k)
            {
                print1(A[i * k + j]);
            }
            else if (lda == m)
            {
                print1(A[j * m + i]);
            }
        }
        printf("\n");
    }
    printf("\n");

    printf("The logical shape of B:\n");
    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (ldb == k)
            {
                print1(B[j * k + i]);
            }
            else if (ldb == n)
            {
                print1(B[i * n + j]);
            }
        }
        printf("\n");
    }
    printf("\n");

    printf("The result of C:\n");
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            printf("%*.0f ", 4, C[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

union GmmaDesc
{
    uint64_t desc_;

    // Bitfield implementation avoids the need for shifts in assignment
    struct
    {
        // start_address, bit [0,14), 4LSB not included
        uint16_t start_address_ : 14, : 2; // 14 bits [0,14), 2 bits unused
        // leading dimension byte offset, bit [16,30), 4LSB not included
        // For N: This is the stride from the first col to the second col of the 8x2 brick in INTERLEAVED
        //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
        // For T: This is the stride from the first 8 rows to the next 8 rows.
        uint16_t leading_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
        // stride dimension byte offset, bit [32,46), 4LSB not included
        // For N: This is the stride from the first 8 rows to the next 8 rows.
        // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
        uint16_t stride_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
        // base_offset, bit [49,52)
        // Valid only for SWIZZLE_128B and SWIZZLE_64B
        uint8_t : 1, base_offset_ : 3, : 4; // 1 bit unused, 3 bits [1,4), 4 bits unused
        // layout type, bit [62,64)
        // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
        uint8_t : 6, layout_type_ : 2; // 6 bits unused, 2 bits [6,8)
    } bitfield;
};

template <class TA, class TB, class TC>
__global__ void wgmma_kernel_m64n32k16_ss_example1(TA *A, TB *B, TC *C, int M, int N, int K)
{
    // 申请矩阵描述符和累加器所需的寄存器
    uint64_t desc_a;
    uint64_t desc_b;
    float d[16] = {0.0f};

    // 设置矩阵描述符参数
    const int scale_D = 0;
    const int scaleA = 1;
    const int scaleB = 1;
    const int tnspA = 0;
    const int tnspB = 0;

    int tid = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    using swizzle_layout = Layout<Shape<_8, _8>, Stride<_8, _1>>; // K-major none swizzle

    // 使用 swizzle atom layout 对 smem 进行 tiling，得到 smem 对应的 layout
    auto sA_layout = tile_to_shape(swizzle_layout{}, make_shape(64, 16));
    auto sB_layout = tile_to_shape(swizzle_layout{}, make_shape(32, 16));

    __shared__ TA smemA[64 * 16];
    auto smemA_cute = make_tensor(make_smem_ptr(smemA), sA_layout); // 根据 smem 的layout 创建对应视图的 tensor

    if (tid < 64)
    {
        for (int j = 0; j < 16; ++j)
        {
            smemA_cute(make_coord(tid, j)) = A[tid * 16 + j]; // 按照 K-major 的方式把数据从 gmem 复制到 smem
        }
    }

    __shared__ TB smemB[32 * 16];
    auto smemB_cute = make_tensor(make_smem_ptr(smemB), sB_layout);
    if (tid < 32)
    {
        for (int j = 0; j < 16; ++j)
        {
            smemB_cute(make_coord(tid, j)) = B[tid * 16 + j];
        }
    }

    __syncthreads();

    GmmaDesc descA;
    descA.bitfield.layout_type_ = 0; // none swizzle
    descA.bitfield.start_address_ = static_cast<uint16_t>(cast_smem_ptr_to_uint(smemA) >> 4);
    descA.bitfield.base_offset_ = 0;
    descA.bitfield.stride_byte_offset_ = 8;   // SBO
    descA.bitfield.leading_byte_offset_ = 64; // LBO
    desc_a = descA.desc_;

    GmmaDesc descB;
    descB.bitfield.layout_type_ = 0;
    descB.bitfield.start_address_ = static_cast<uint16_t>(cast_smem_ptr_to_uint(smemB) >> 4);
    descB.bitfield.base_offset_ = 0;
    descB.bitfield.stride_byte_offset_ = 8;
    descB.bitfield.leading_byte_offset_ = 32;
    desc_b = descB.desc_;

    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " p,   %19, %20, %21, %22;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(desc_a),
          "l"(desc_b),
          "r"(int32_t(scale_D)), "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));

    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

    __syncthreads();

    int row_c = lane_id / 4 + warp_id * 16;
    int col_c = lane_id % 4 * 2;

    for (int i = 0; i < 4; i++)
    {
        C[row_c * N + col_c + 8 * i + 0] = d[i * 4 + 0];
        C[row_c * N + col_c + 8 * i + 1] = d[i * 4 + 1];
        C[(row_c + 8) * N + col_c + 8 * i + 0] = d[i * 4 + 2];
        C[(row_c + 8) * N + col_c + 8 * i + 1] = d[i * 4 + 3];
    }
}

template <class TA, class TB, class TC>
__global__ void wgmma_kernel_m64n32k16_ss_example2(TA *A, TB *B, TC *C, int M, int N, int K)
{
    // 申请矩阵描述符和累加器所需的寄存器
    uint64_t desc_a;
    uint64_t desc_b;
    float d[16] = {0.0f};

    // 设置矩阵描述符参数
    const int scale_D = 0;
    const int scaleA = 1;
    const int scaleB = 1;
    const int tnspA = 1;
    const int tnspB = 1;

    int tid = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    using swizzle_layoutA = ComposedLayout<Swizzle<2, 3, 3>, _0, Layout<Shape<_32, _8>, Stride<_1, _32>>>; // MN-major 64B swizzle
    using swizzle_layoutB = Layout<Shape<_8, _8>, Stride<_1, _8>>;                                         // MN-major none swizzle

    // 使用 swizzle atom layout 对 smem 进行 tiling，得到 smem 对应的 layout
    auto sA_layout = tile_to_shape(swizzle_layoutA{}, make_shape(64, 16));
    auto sB_layout = tile_to_shape(swizzle_layoutB{}, make_shape(32, 16));

    __shared__ TA smemA[64 * 16];
    auto smemA_cute = make_tensor(make_smem_ptr(smemA), sA_layout); // 根据 smem 的layout 创建对应视图的 tensor

    if (tid < 64)
    {
        for (int j = 0; j < 16; ++j)
        {
            smemA_cute(make_coord(tid, j)) = A[tid + j * 64]; // 按照 M-major 的方式把数据从 gmem 复制到 smem
        }
    }

    __shared__ TB smemB[32 * 16];
    auto smemB_cute = make_tensor(make_smem_ptr(smemB), sB_layout);
    if (tid < 32)
    {
        for (int j = 0; j < 16; ++j)
        {
            smemB_cute(make_coord(tid, j)) = B[tid + j * 32];
        }
    }

    __syncthreads();

    GmmaDesc descA;
    descA.bitfield.layout_type_ = 2; // 64B swizzle
    descA.bitfield.start_address_ = static_cast<uint16_t>(cast_smem_ptr_to_uint(smemA) >> 4);
    descA.bitfield.base_offset_ = 0;
    descA.bitfield.stride_byte_offset_ = 64;
    descA.bitfield.leading_byte_offset_ = 32;
    desc_a = descA.desc_;

    GmmaDesc descB;
    descB.bitfield.layout_type_ = 0;
    descB.bitfield.start_address_ = static_cast<uint16_t>(cast_smem_ptr_to_uint(smemB) >> 4);
    descB.bitfield.base_offset_ = 0;
    descB.bitfield.stride_byte_offset_ = 8;
    descB.bitfield.leading_byte_offset_ = 32;
    desc_b = descB.desc_;

    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " p,   %19, %20, %21, %22;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(desc_a),
          "l"(desc_b),
          "r"(int32_t(scale_D)), "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));

    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

    __syncthreads();

    int row_c = lane_id / 4 + warp_id * 16;
    int col_c = lane_id % 4 * 2;

    for (int i = 0; i < 4; i++)
    {
        C[row_c * N + col_c + 8 * i + 0] = d[i * 4 + 0];
        C[row_c * N + col_c + 8 * i + 1] = d[i * 4 + 1];
        C[(row_c + 8) * N + col_c + 8 * i + 0] = d[i * 4 + 2];
        C[(row_c + 8) * N + col_c + 8 * i + 1] = d[i * 4 + 3];
    }
}

template <class TA, class TB, class TC>
__global__ void wgmma_kernel_m64n32k16_ss_example3(TA *A, TB *B, TC *C, int M, int N, int K)
{
    // 申请矩阵描述符和累加器所需的寄存器
    uint64_t desc_a;
    uint64_t desc_b;
    float d[16] = {0.0f};

    // 设置矩阵描述符参数
    const int scale_D = 0;
    const int scaleA = 1;
    const int scaleB = 1;
    const int tnspA = 1;
    const int tnspB = 0;

    int tid = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    using swizzle_layoutA = ComposedLayout<Swizzle<3, 3, 3>, _0, Layout<Shape<_64, _8>, Stride<_1, _64>>>; // MN-major 128B swizzle
    using swizzle_layoutB = ComposedLayout<Swizzle<1, 3, 3>, _0, Layout<Shape<_8, _16>, Stride<_16, _1>>>; // K-major 32B swizzle

    // 使用 swizzle atom layout 对 smem 进行 tiling，得到 smem 对应的 layout
    auto sA_layout = tile_to_shape(swizzle_layoutA{}, make_shape(64, 16));
    auto sB_layout = tile_to_shape(swizzle_layoutB{}, make_shape(32, 16));

    __shared__ TA smemA[64 * 16];
    auto smemA_cute = make_tensor(make_smem_ptr(smemA), sA_layout); // 根据 smem 的layout 创建对应视图的 tensor

    if (tid < 64)
    {
        for (int j = 0; j < 16; ++j)
        {
            smemA_cute(make_coord(tid, j)) = A[tid + j * 64]; // 按照 M-major 的方式把数据从 gmem 复制到 smem
        }
    }

    __shared__ TB smemB[32 * 16];
    auto smemB_cute = make_tensor(make_smem_ptr(smemB), sB_layout);
    if (tid < 32)
    {
        for (int j = 0; j < 16; ++j)
        {
            smemB_cute(make_coord(tid, j)) = B[tid * 16 + j]; // 按照 K-major 的方式把数据从 gmem 复制到 smem
        }
    }

    __syncthreads();

    GmmaDesc descA;
    descA.bitfield.layout_type_ = 1; // 128B swizzle
    descA.bitfield.start_address_ = static_cast<uint16_t>(cast_smem_ptr_to_uint(smemA) >> 4);
    descA.bitfield.base_offset_ = 0;
    descA.bitfield.stride_byte_offset_ = 64;  // SBO
    descA.bitfield.leading_byte_offset_ = 64; // LBO 这里因为在 LBO 方向上只重复一次，所以可以随便设
    desc_a = descA.desc_;

    GmmaDesc descB;
    descB.bitfield.layout_type_ = 3; // 32B swizzle
    descB.bitfield.start_address_ = static_cast<uint16_t>(cast_smem_ptr_to_uint(smemB) >> 4);
    descB.bitfield.base_offset_ = 0;
    descB.bitfield.stride_byte_offset_ = 16;
    descB.bitfield.leading_byte_offset_ = 1;
    desc_b = descB.desc_;

    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " p,   %19, %20, %21, %22;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(desc_a),
          "l"(desc_b),
          "r"(int32_t(scale_D)), "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));

    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

    __syncthreads();

    int row_c = lane_id / 4 + warp_id * 16;
    int col_c = lane_id % 4 * 2;

    for (int i = 0; i < 4; i++)
    {
        C[row_c * N + col_c + 8 * i + 0] = d[i * 4 + 0];
        C[row_c * N + col_c + 8 * i + 1] = d[i * 4 + 1];
        C[(row_c + 8) * N + col_c + 8 * i + 0] = d[i * 4 + 2];
        C[(row_c + 8) * N + col_c + 8 * i + 1] = d[i * 4 + 3];
    }
}

template <class TA, class TB, class TC>
__global__ void wgmma_kernel_m64n32k16_rs_example4(TA *A, TB *B, TC *C, int M, int N, int K)
{
    uint32_t a[4];
    uint64_t desc_b;
    float d[16] = {0.0f};

    const int scale_D = 0;
    const int scaleA = 1;
    const int scaleB = 1;
    const int tnspB = 1;

    int tid = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int row_a = lane_id / 4 + warp_id * 16;
    int col_a = lane_id % 4 * 2;
    a[0] = *reinterpret_cast<uint32_t *>(A + row_a * 16 + col_a);
    a[1] = *reinterpret_cast<uint32_t *>(A + (row_a + 8) * 16 + col_a);
    a[2] = *reinterpret_cast<uint32_t *>(A + row_a * 16 + col_a + 8);
    a[3] = *reinterpret_cast<uint32_t *>(A + (row_a + 8) * 16 + col_a + 8);

    using swizzle_layoutB = ComposedLayout<Swizzle<2, 3, 3>, _0, Layout<Shape<_32, _8>, Stride<_1, _32>>>; // MN-major 64B swizzle

    // 使用 swizzle atom layout 对 smem 进行 tiling，得到 smem 对应的 layout
    auto sB_layout = tile_to_shape(swizzle_layoutB{}, make_shape(32, 16));

    __shared__ TB smemB[32 * 16];
    auto smemB_cute = make_tensor(make_smem_ptr(smemB), sB_layout);
    if (tid < 32)
    {
        for (int j = 0; j < 16; ++j)
        {
            smemB_cute(make_coord(tid, j)) = B[tid + j * 32];
        }
    }
    __syncthreads();

    GmmaDesc descB;
    descB.bitfield.layout_type_ = 2;
    descB.bitfield.start_address_ = static_cast<uint16_t>(cast_smem_ptr_to_uint(smemB) >> 4);
    descB.bitfield.base_offset_ = 0;
    descB.bitfield.stride_byte_offset_ = 32;
    descB.bitfield.leading_byte_offset_ = 1; // LBO，因为只重复一次，所以可以随便设
    desc_b = descB.desc_;

    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,"
        " p,   %38, %39, %40;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(desc_b),
          "r"(int32_t(scale_D)), "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspB)));

    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

    __syncthreads();

    for (int i = 0; i < 8; i++)
    {
        C[row_a * N + col_a + 8 * i + 0] = d[i * 4 + 0];
        C[row_a * N + col_a + 8 * i + 1] = d[i * 4 + 1];
        C[(row_a + 8) * N + col_a + 8 * i + 0] = d[i * 4 + 2];
        C[(row_a + 8) * N + col_a + 8 * i + 1] = d[i * 4 + 3];
    }
}

int main()
{
    srand(1234);

    int M = 64, N = 32, K = 16;

    int A_size = M * K;
    int B_size = K * N;
    int C_size = M * N;

    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = float;

    thrust::host_vector<TA> h_A(A_size);
    thrust::host_vector<TB> h_B(B_size);
    thrust::host_vector<TC> h_C(C_size);

    for (int i = 0; i < A_size; ++i)
    {
        h_A[i] = static_cast<TA>(i);
    }
    for (int i = 0; i < B_size; ++i)
    {
        h_B[i] = static_cast<TB>(i);
    }

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    thrust::host_vector<TC> cute_result;

    dim3 blocks(1);
    dim3 threads(128);

    // A K-major none swizzle, B K-major none swizzle
    wgmma_kernel_m64n32k16_ss_example1<TA, TB, TC><<<blocks, threads>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    cute_result = d_C;
    print_matrix(h_A.data(), h_B.data(), cute_result.data(), M, N, K, K, K);

    // A M-major 64B swizzle, B N-major none swizzle
    wgmma_kernel_m64n32k16_ss_example2<TA, TB, TC><<<blocks, threads>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    cute_result = d_C;
    print_matrix(h_A.data(), h_B.data(), cute_result.data(), M, N, K, M, N);

    // A M-major 128B swizzle, B K-major 32B swizzle
    wgmma_kernel_m64n32k16_ss_example3<TA, TB, TC><<<blocks, threads>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    cute_result = d_C;
    print_matrix(h_A.data(), h_B.data(), cute_result.data(), M, N, K, M, K);

    // A K-major registers, B N-major 64B swizzle
    wgmma_kernel_m64n32k16_rs_example4<TA, TB, TC><<<blocks, threads>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
    cute_result = d_C;
    print_matrix(h_A.data(), h_B.data(), cute_result.data(), M, N, K, K, N);

    return 0;
}
