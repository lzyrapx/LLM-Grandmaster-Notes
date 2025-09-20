#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
// #include <cute/tensor.hpp>

// using namespace cute;

#include <cuda/barrier>
#include <cuda/ptx>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;
namespace cde = cuda::device::experimental;

__device__ uint32_t cast_smem_ptr_to_uint(void const *const ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__global__ void cp_async_size4(float *src, float *dst, int N)
{
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    extern __shared__ float smem[];

    float *gmem_ptr = src + index;
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + tid));

    // 启动一个异步拷贝，一次拷贝 4 bytes
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr),
                 "n"(sizeof(float)));

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);

    dst[index] = smem[tid];
}

__global__ void cp_async_size16(float *src, float *dst, int N)
{
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x * 4;

    extern __shared__ float smem[];

    float *gmem_ptr = src + index + tid * 4;
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<float4 *>(smem) + tid));

    // 启动一个异步拷贝，一次拷贝 16 bytes
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr),
                 "n"(sizeof(float4)));

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);

    reinterpret_cast<float4 *>(dst + index)[tid] = reinterpret_cast<float4 *>(smem)[tid];
}

__global__ void cp_async_bulk(float *src, float *dst, int N)
{
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x;

    extern __shared__ float smem[];
    __shared__ uint64_t bar[1];

    int transaction_bytes = blockDim.x * sizeof(float);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(bar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem);

    if (tid == 0)
    {
        /// Initialize shared memory barrier
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_mbar),
                     "r"(blockDim.x));
        // asm volatile ("fence.proxy.async.shared::cta;");
        asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_mbar),
                     "r"(transaction_bytes));
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :
                     : "r"(smem_int_ptr), "l"(src + index), "r"(transaction_bytes), "r"(smem_int_mbar)
                     : "memory");
    }
    __syncthreads();

    // arrive
    uint64_t token = 0;
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n" ::"l"(token), "r"(smem_int_mbar));

    // wait
    asm volatile(
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra DONE;\n"
        "bra                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(smem_int_mbar),
        "l"(token));

    // compute

    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();

    // store, shared memory to global memory
    if (tid == 0)
    {
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                     :
                     : "l"(dst + index), "r"(smem_int_ptr), "r"(transaction_bytes)
                     : "memory");
        asm volatile("cp.async.bulk.commit_group;");
        asm volatile(
            "cp.async.bulk.wait_group.read %0;"
            :
            : "n"(0)
            : "memory");
    }
}

template <uint32_t RANK>
CUtensorMap make_gemm_tma_desc(void *gmem_tensor_ptr, std::vector<int> &gmem_shape, std::vector<int> &smem_shape)
{
    CUtensorMap tensor_map{};

    uint64_t gmem_prob_shape[5] = {1, 1, 1, 1, 1};
    uint64_t gmem_prob_stride[5] = {0, 0, 0, 0, 0};
    uint32_t smem_box_shape[5] = {1, 1, 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    gmem_prob_shape[0] = gmem_shape[0];
    gmem_prob_stride[0] = 1;
    smem_box_shape[0] = smem_shape[0];

    for (int i = 1; i < RANK; ++i)
    {
        gmem_prob_shape[i] = gmem_shape[i];
        gmem_prob_stride[i] = gmem_prob_stride[i - 1] * gmem_shape[i - 1];
        smem_box_shape[i] = smem_shape[i];
    }

    for (int i = 0; i < RANK; ++i)
    {
        gmem_prob_stride[i] *= sizeof(float);
    }

    auto tma_format = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    auto tma_interleave = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE;
    auto smem_swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    auto tma_l2Promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE;
    auto tma_oobFill = CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    // Create the tensor descriptor.
    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map, // CUtensorMap *tensorMap,
        tma_format,
        RANK,                 // cuuint32_t tensorRank,
        gmem_tensor_ptr,      // void *globalAddress,
        gmem_prob_shape,      // const cuuint64_t *globalDim,
        gmem_prob_stride + 1, // const cuuint64_t *globalStrides,
        smem_box_shape,       // const cuuint32_t *boxDim,
        smem_box_stride,      // const cuuint32_t *elementStrides,
        tma_interleave,       // Interleave patterns can be used to accelerate loading of values that are less than 4 bytes long.
        smem_swizzle,         // Swizzling can be used to avoid shared memory bank conflicts.
        tma_l2Promotion,      // L2 Promotion can be used to widen the effect of a cache-policy to a wider set of L2 cache lines.
        tma_oobFill           // Any element that is outside of bounds will be set to zero by the TMA transfer.
    );

    if (result != CUDA_SUCCESS)
    {
        std::cerr << "TMA Desc Addr:   " << &tensor_map
                  << "\nformat         " << tma_format
                  << "\ndim            " << RANK
                  << "\ngmem_address   " << gmem_tensor_ptr
                  << "\nglobalDim      " << gmem_prob_shape
                  << "\nglobalStrides  " << gmem_prob_stride
                  << "\nboxDim         " << smem_box_shape
                  << "\nelementStrides " << smem_box_stride
                  << "\ninterleave     " << tma_interleave
                  << "\nswizzle        " << smem_swizzle
                  << "\nl2Promotion    " << tma_l2Promotion
                  << "\noobFill        " << tma_oobFill << std::endl;
        std::cerr << "Error: Failed to initialize the TMA descriptor " << result << std::endl;
        assert(false);
    }
    else
    {
        printf("tma desc success\n");
    }

    return tensor_map;
}

__global__ void cp_async_bulk_tensor_1d(const __grid_constant__ CUtensorMap src_tensor_map, const __grid_constant__ CUtensorMap dst_tensor_map)
{
    int tid = threadIdx.x;
    int crd0 = blockIdx.x * 256;
    __shared__ alignas(128) float smem[256]; // 256 float
    __shared__ alignas(8) uint64_t bar[1];

    int transaction_bytes = blockDim.x * sizeof(float);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(bar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem);

    if (tid == 0)
    {
        /// Initialize shared memory barrier
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_mbar),
                     "r"(blockDim.x));
        asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_mbar),
                     "r"(transaction_bytes));
        asm volatile("fence.proxy.async.shared::cta;");
        asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%3}], [%2];" ::"r"(smem_int_ptr),
                     "l"(&src_tensor_map), "r"(smem_int_mbar), "r"(crd0) : "memory");
    }
    __syncthreads();

    // arrive
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_int_mbar));

    // wait
    int phase_bit = 0;
    asm volatile(
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra DONE;\n"
        "bra                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(smem_int_mbar),
        "r"(phase_bit));

    // compute

    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();

    // store shared memory to global memory
    if (tid == 0)
    {
        asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%0, {%2}], [%1];" ::"l"(&dst_tensor_map), "r"(smem_int_ptr), "r"(crd0) : "memory");
        asm volatile("cp.async.bulk.commit_group;");
        asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(0) : "memory");
    }
}

__global__ void cp_async_bulk_tensor_2d(const __grid_constant__ CUtensorMap src_tensor_map, const __grid_constant__ CUtensorMap dst_tensor_map)
{
    int tid = threadIdx.x;
    int crd0 = blockIdx.x % 128 * 8;
    int crd1 = blockIdx.x / 128 * 16;

    __shared__ alignas(128) float smem[128];
    __shared__ alignas(8) uint64_t bar[1];

    int transaction_bytes = 128 * sizeof(float);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(bar);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem);

    if (tid == 0)
    {
        /// Initialize shared memory barrier
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_mbar),
                     "r"(blockDim.x));
        asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_mbar),
                     "r"(transaction_bytes));
        asm volatile("fence.proxy.async.shared::cta;");
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_int_ptr),
                     "l"(&src_tensor_map), "r"(smem_int_mbar), "r"(crd0), "r"(crd1) : "memory");
    }
    __syncthreads();

    // arrive
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_int_mbar));

    // wait
    int phase_bit = 0;
    asm volatile(
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra DONE;\n"
        "bra                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(smem_int_mbar),
        "r"(phase_bit));

    // compute
    if (tid == 0 && blockIdx.x == 0)
    {
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 32; ++j)
            {
                printf("%.0f ", smem[i * 32 + j]);
            }
            printf("\n");
        }
    }

    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();

    // store shared memory to global memory
    if (tid == 0)
    {
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" ::"l"(&dst_tensor_map), "r"(smem_int_ptr), "r"(crd0), "r"(crd1) : "memory");
        asm volatile("cp.async.bulk.commit_group;");
        asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(0) : "memory");
    }
}

void test_copy(float *src, float *dst, int N)
{
    for (int i = 0; i < N; ++i)
    {
        // printf("idx = %d, src = %f, dst = %f\n", i, src[i], dst[i]);
        if (src[i] != dst[i])
        {
            printf("copy failed\n");
            return;
        }
    }
    printf("copy success\n");
}

// nvcc cp_async.cu -o cpasync -arch=sm_90a -lcuda -I ../../include/ -I ../../tools/util/include/ -std=c++17
int main()
{
    srand(1234);

    int N = 1024 * 1024;

    thrust::host_vector<float> h_S(N);
    thrust::host_vector<float> h_D(N);
    thrust::host_vector<float> copy_result(N);

    for (int i = 0; i < N; ++i)
    {
        h_S[i] = static_cast<float>(i % 1024);
    }

    thrust::device_vector<float> d_S = h_S;
    thrust::device_vector<float> d_D = h_D;

    std::vector<int> gmem_shape = {1024, 1024};
    std::vector<int> smem_shape = {8, 16};

    auto src_gmem_desc = make_gemm_tma_desc<2>(d_S.data().get(), gmem_shape, smem_shape);
    auto dst_gmem_desc = make_gemm_tma_desc<2>(d_D.data().get(), gmem_shape, smem_shape);

    constexpr int threads = 128;
    int blocks = (N + threads - 1) / threads;

    // cp_async_size4<<<blocks, threads, threads * sizeof(float)>>>(d_S.data().get(), d_D.data().get(), N);
    // cp_async_size16<<<blocks, threads, threads * sizeof(float4)>>>(d_S.data().get(), d_D.data().get(), N);

    // cp_async_bulk_tensor_1d<<<blocks, threads>>>(src_gmem_desc, dst_gmem_desc);
    cp_async_bulk_tensor_2d<<<blocks, threads>>>(src_gmem_desc, dst_gmem_desc);

    copy_result = d_D;
    test_copy(h_S.data(), copy_result.data(), N);

    return 0;
}
