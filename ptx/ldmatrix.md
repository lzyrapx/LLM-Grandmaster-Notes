# ldmatrix

## 指令介绍
ldmatrix是一个可以把数据从共享内存加载到寄存器中的指令，用于配合mma使用。ldmatrix也是一个warp-level的指令，一个warp中的32个线程合作搬运数据。
ldmatrix指令的使用方式如下：

```cpp
ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type r, [p];

ldmatrix.sync.aligned.m8n16.num{.ss}.dst_fmt.src_fmt        r, [p];
ldmatrix.sync.aligned.m16n16.num.trans{.ss}.dst_fmt.src_fmt r, [p];

.shape   = {.m8n8, .m16n16};
.num     = {.x1, .x2, .x4};
.ss      = {.shared{::cta}};
.type    = {.b16, .b8};
.dst_fmt = { .b8x16 };
.src_fmt = { .b6x16_p32, .b4x16_p64 };
```

指令中.sync和.aligned的含义与mma中的相同，.sync代表ldmatrix会使执行线程等待，直到 warp 中的所有线程都执行相同的 ldmatrix 指令后才会继续执行。.aligned 限定符表示，warp 中的所有线程必须执行相同的 ldmatrix 指令。
.shape代表一个指令加载的矩阵大小，根据不同的数据类型支持m8n8，m16n16和m8n16大小的矩阵。shape与数据类型的对应关系如下表所示。

| shape   | Matrix shape | Element size               |
|:-------:|:------------:|:--------------------------:|
| .m8n8   | 8x8          | 16-bit                     |
| .m16n16 | 16x16        | 8-bit or 6-bit or 4-bit    |
| .m8n16  | 8x16         | 6-bit or 4-bit             |

从上表可以看到，ldmatrix支持16bit，8bit，6bit和4bit，type里的b16代表16-bit的意思，不是bf16，因此可以处理8×8大小的bf16或fp16的数据，也可以处理8×4大小的32位数据。
使用ldmatrix指令处理16bit的数据需要sm_75以上，加载其他的数据类型则需要sm_100a以上，因此后面只讨论m8n8大小数据的处理方式。
.num可以是.x1，.x2和.x4，分别代表加载1个，2个和4个8×8大小的矩阵。

## 使用方法

一个8×8大小的矩阵中每一行不需要在内存中连续存储，但是一行中的8个元素需要连续。
每个矩阵所需的八个地址由八个线程提供，具体取决于.num的值，如下表所示。每个地址对应矩阵行的起始位置。地址addr0到addr7对应第一个矩阵的行，地址addr8到addr15对应第二个矩阵的行，依此类推。

| .num | Threads 0-7    | Threads 8-15   | Threads 16-23  | Threads 24-31  |
|------|---------------|---------------|---------------|---------------|
| .x1  | addr0-addr7   | -             | -             | -             |
| .x2  | addr0-addr7   | addr8-addr15  | -             | -             |
| .x4  | addr0-addr7   | addr8-addr15  | addr16-addr23 | addr24-addr31 |

当num=x1时ldmatrix加载一个8*8矩阵中的64个16bit数据。只需要将8行中每行的首地址传给0-7号线程即可，其余线程不需要处理地址。一个warp中32个线程的每一个线程会用一个32bit的寄存器加载2个16bit的数据，所以一共会加载64个数据。
(TODO:是0-7一个线程读取8个元素然后分发到32个线程中，还是32个线程分别读取2个元素。感觉前者合理点)
一个线程读取128bits的数据，然后再分发到32个线程中。证据是当ldmatrix的num=x1时，一个线程加载了128bits的数据。如果是后者的话，一个线程只会加载32bits的数据。

每个线程和对应加载的数据如下图所示。每一行用4个线程加载一行中的8个数据。
插个图

当使用可选的限定符.trans时，表示矩阵以列主序（column-major）格式加载。线程与元素的对应关系会变成下面这样。
插个图

上面是num=x1的情况，num=x2和num=x4的情况基本相同。
当num=x2时，ldmatrix会加载2个8*8矩阵中的128个16bit数据。此时需要将16行中每行的首地址传给0-7和8-15号线程，其余线程不需要处理地址。一个warp中的32个线程中每一个线程都会使用2个32bit的寄存器加载4个16bit的数据，一共会加载两个矩阵中的128个数据。当num=x2时第二个矩阵的元素会按照上表中的布局加载到每个线程的第二个寄存器中。每个线程和元素的对应关系如下。
插个图

当num=x4时，ldmatrix会加载4个8*8矩阵中的256个16bit数据。此时需要将32个行的首地址分别传给32个线程。一个线程需要使用4个32bit寄存器加载8个16bit的数据，所以可以加载4个8*8矩阵中的256个数据。加载时也是以8*8大小为单位，按照num=x1的布局，一个线程分别加载四个8*8矩阵中的2个数据，正好可以加载4个矩阵中的8个元素。线程与元素的对应关系如下。


## 代码测试
下面使用代码来测试ldmatrix是如何从共享内存中加载数据的。

### num=x1
当num=x1时可以处理一个8*8矩阵。首先将数据从全局内存加载到共享内存中。TS代表src_data的数据类型，在这里时fp16。

```cpp
    int tid = threadIdx.x;

    __shared__ TS smem_src[64];
    smem_src[tid] = S[tid];
    smem_src[tid + 32] = S[tid + 32];

    __syncthreads();
```

然后处理每行的首地址和线程的对应关系。根据上面的介绍，num=x1时只需要0-7号线程来处理8行的首地址就行了，其余线程的地址不会对结果产生影响，所以可以写成下面这样。
```cpp
    TS *smem_addr = nullptr;
    if (tid < 8)
    {
        smem_addr = smem_src + tid % 8 * N;
    }
```

然后使用ldmatrix指令进行搬运数据。这里每个线程对应的数据会保存在dst1中。

```cpp
    uint32_t dst1;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_addr);
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
                 : "=r"(dst1)
                 : "r"(smem_int_ptr));
```

到这里32个线程分别加载了8*8矩阵中的2个元素。为了验证加载结果，我们把每个线程寄存器中的数据保存到D中。

```cpp
    half_t r1, r2;
    r1.storage = static_cast<uint16_t>(dst1 & 0xFFFF);
    r2.storage = static_cast<uint16_t>(dst1 >> 16);

    int row = tid / 4;
    int col = tid % 4 * 2;
    D[row * N + col + 0] = r1;
    D[row * N + col + 1] = r2;
```

打印结果如下。
```cpp
thread=0, val= 1  3
thread=1, val= 1  4
thread=2, val= 5  4
thread=3, val= 8  9
thread=4, val= 6  5
thread=5, val= 7  7
thread=6, val= 2  3
thread=7, val= 1  6
thread=8, val= 6  5
thread=9, val= 8  4
thread=10, val= 3  9
thread=11, val= 9  2
thread=12, val= 8  4
thread=13, val= 2  8
thread=14, val= 7  9
thread=15, val= 3  7
thread=16, val= 2  1
thread=17, val= 9  4
thread=18, val= 4  7
thread=19, val= 3  7
thread=20, val= 9  1
thread=21, val= 5  8
thread=22, val= 3  3
thread=23, val= 4  8
thread=24, val= 5  1
thread=25, val= 3  8
thread=26, val= 9  9
thread=27, val= 7  5
thread=28, val= 1  6
thread=29, val= 1  8
thread=30, val= 5  4
thread=31, val= 3  4
m = 8, n = 8
The logical shape of A:
 1  3  1  4  5  4  8  9
 6  5  7  7  2  3  1  6
 6  5  8  4  3  9  9  2
 8  4  2  8  7  9  3  7
 2  1  9  4  4  7  3  7
 9  1  5  8  3  3  4  8
 5  1  3  8  9  9  7  5
 1  6  1  8  5  4  3  4

The copy result of B:
 1  3  1  4  5  4  8  9
 6  5  7  7  2  3  1  6
 6  5  8  4  3  9  9  2
 8  4  2  8  7  9  3  7
 2  1  9  4  4  7  3  7
 9  1  5  8  3  3  4  8
 5  1  3  8  9  9  7  5
 1  6  1  8  5  4  3  4
```

完整代码

```cpp
template <class TS, class TD>
__global__ void ldmatrix_x1(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;

    __shared__ TS smem_src[64];
    smem_src[tid] = S[tid];
    smem_src[tid + 32] = S[tid + 32];

    __syncthreads();

    TS *smem_addr = nullptr;
    if (tid < 8)
    {
        smem_addr = smem_src + tid % 8 * N;
    }

    uint32_t dst1;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_addr);
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
                 : "=r"(dst1)
                 : "r"(smem_int_ptr));

    half_t r1, r2;
    r1.storage = static_cast<uint16_t>(dst1 & 0xFFFF);
    r2.storage = static_cast<uint16_t>(dst1 >> 16);

    print_value(tid, r1, r2);

    int row = tid / 4;
    int col = tid % 4 * 2;
    D[row * N + col + 0] = r1;
    D[row * N + col + 1] = r2;
}
```

### num=x2
```cpp
template <class TS, class TD>
__global__ void ldmatrix_x2(TS *S, TD *D, int M, int N)
{
    __shared__ TS smem_src[128];
    for (int i = 0; i < M * N; ++i)
    {
        smem_src[i] = S[i];
    }
    __syncthreads();

    int tid = threadIdx.x;

    TS *smem_addr = nullptr;
    if (tid < 16)
    {
        smem_addr = smem_src + tid % M * N;
    }

    uint32_t dst0, dst1;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_addr);
    asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(smem_int_ptr));

    half_t r1, r2, r3, r4;
    r1.storage = static_cast<uint16_t>(dst0 & 0xFFFF);
    r2.storage = static_cast<uint16_t>(dst0 >> 16);
    r3.storage = static_cast<uint16_t>(dst1 & 0xFFFF);
    r4.storage = static_cast<uint16_t>(dst1 >> 16);

    int row = tid / 4;
    int col = tid % 4 * 2;
    D[row * N + col + 0] = r1;
    D[row * N + col + 1] = r2;
    D[(row + 8) * N + col + 0] = r3;
    D[(row + 8) * N + col + 1] = r4;
}
```

输出结果：

```cpp
m = 16, n = 8
The logical shape of A:
 1  3  1  4  5  4  8  9
 6  5  7  7  2  3  1  6
 6  5  8  4  3  9  9  2
 8  4  2  8  7  9  3  7
 2  1  9  4  4  7  3  7
 9  1  5  8  3  3  4  8
 5  1  3  8  9  9  7  5
 1  6  1  8  5  4  3  4
 2  9  8  6  4  1  1  2
 1  3  9  2  3  4  7  8
 4  9  4  1  7  8  5  7
 2  5  3  4  6  4  7  8
 3  5  2  5  4  9  6  2
 3  5  3  3  8  1  1  9
 7  4  9  2  9  2  9  1
 7  9  4  1  3  2  8  4

The logical shape of B:
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0

The copy result of B:
 1  3  1  4  5  4  8  9
 6  5  7  7  2  3  1  6
 6  5  8  4  3  9  9  2
 8  4  2  8  7  9  3  7
 2  1  9  4  4  7  3  7
 9  1  5  8  3  3  4  8
 5  1  3  8  9  9  7  5
 1  6  1  8  5  4  3  4
 2  9  8  6  4  1  1  2
 1  3  9  2  3  4  7  8
 4  9  4  1  7  8  5  7
 2  5  3  4  6  4  7  8
 3  5  2  5  4  9  6  2
 3  5  3  3  8  1  1  9
 7  4  9  2  9  2  9  1
 7  9  4  1  3  2  8  4
```

### num=x4

num=x4基本包含了num=x2，所以直接介绍num=x4时ldmatrix的搬运过程。
num=x4时可以处理4个8*8矩阵，因此一共可以处理256个16bit元素。同样的，我们需要先把数据搬运到共享内存中。

```cpp
    int tid = threadIdx.x;

    __shared__ TS smem_src[256];
    for (int i = 0; i < 8; ++i)
    {
        smemA[tid + i * 32] = A[tid + i * 32];
    }
    __syncthreads();
```

然后处理每个线程与每行矩阵首地址的对应关系。此时我们需要使用全部32个线程处理32行的首地址。

```cpp
TS *smem_addr = smem_src + tid % M * N + tid / M * 8;
```

然后使用ldmatrix进行加载。此时一个线程有4个寄存器，可以加载8个16bit数据。
```cpp
    uint32_t dst0, dst1, dst2, dst3;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_addr);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
                 : "r"(smem_int_ptr));
```

打印一下每个线程对应的元素和搬运结果。
```cpp
thread=0, val=1 3 4 7 6 5 6 9
thread=1, val=1 4 8 5 7 7 9 3
thread=2, val=5 4 7 4 2 3 7 1
thread=3, val=8 9 7 7 1 6 2 3
thread=4, val=6 5 2 9 8 4 1 4
thread=5, val=8 4 4 2 2 8 6 3
thread=6, val=3 9 1 3 7 9 5 2
thread=7, val=9 2 2 5 3 7 4 6
thread=8, val=2 1 7 9 9 1 5 7
thread=9, val=9 4 2 4 5 8 9 1
thread=10, val=4 7 3 8 3 3 5 1
thread=11, val=3 7 1 7 4 8 3 7
thread=12, val=5 1 7 6 1 6 8 4
thread=13, val=3 8 6 8 1 8 9 3
thread=14, val=9 9 9 5 5 4 4 2
thread=15, val=7 5 1 7 3 4 7 1
thread=16, val=2 9 1 8 1 3 6 7
thread=17, val=8 6 4 2 9 2 6 2
thread=18, val=4 1 4 2 3 4 5 8
thread=19, val=1 2 8 6 7 8 8 3
thread=20, val=4 9 5 2 2 5 1 5
thread=21, val=4 1 8 2 3 4 5 4
thread=22, val=7 8 6 8 6 4 4 2
thread=23, val=5 7 8 5 7 8 4 5
thread=24, val=3 5 7 5 3 5 6 3
thread=25, val=2 5 4 1 3 3 3 8
thread=26, val=4 9 6 2 8 1 9 8
thread=27, val=6 2 7 2 1 9 1 4
thread=28, val=7 4 9 6 7 9 1 3
thread=29, val=9 2 3 6 4 1 4 5
thread=30, val=9 2 3 8 3 2 3 7
thread=31, val=9 1 8 3 8 4 9 9
m = 16, n = 16
The logical shape of A:
 1  3  1  4  5  4  8  9  6  5  7  7  2  3  1  6
 6  5  8  4  3  9  9  2  8  4  2  8  7  9  3  7
 2  1  9  4  4  7  3  7  9  1  5  8  3  3  4  8
 5  1  3  8  9  9  7  5  1  6  1  8  5  4  3  4
 2  9  8  6  4  1  1  2  1  3  9  2  3  4  7  8
 4  9  4  1  7  8  5  7  2  5  3  4  6  4  7  8
 3  5  2  5  4  9  6  2  3  5  3  3  8  1  1  9
 7  4  9  2  9  2  9  1  7  9  4  1  3  2  8  4
 4  7  8  5  7  4  7  7  6  9  9  3  7  1  2  3
 2  9  4  2  1  3  2  5  1  4  6  3  5  2  4  6
 7  9  2  4  3  8  1  7  5  7  9  1  5  1  3  7
 7  6  6  8  9  5  1  7  8  4  9  3  4  2  7  1
 1  8  4  2  4  2  8  6  6  7  6  2  5  8  8  3
 5  2  8  2  6  8  8  5  1  5  5  4  4  2  4  5
 7  5  4  1  6  2  7  2  6  3  3  8  9  8  1  4
 9  6  3  6  3  8  8  3  1  3  4  5  3  7  9  9

The copy result of B:
 1  3  1  4  5  4  8  9  6  5  7  7  2  3  1  6
 6  5  8  4  3  9  9  2  8  4  2  8  7  9  3  7
 2  1  9  4  4  7  3  7  9  1  5  8  3  3  4  8
 5  1  3  8  9  9  7  5  1  6  1  8  5  4  3  4
 2  9  8  6  4  1  1  2  1  3  9  2  3  4  7  8
 4  9  4  1  7  8  5  7  2  5  3  4  6  4  7  8
 3  5  2  5  4  9  6  2  3  5  3  3  8  1  1  9
 7  4  9  2  9  2  9  1  7  9  4  1  3  2  8  4
 4  7  8  5  7  4  7  7  6  9  9  3  7  1  2  3
 2  9  4  2  1  3  2  5  1  4  6  3  5  2  4  6
 7  9  2  4  3  8  1  7  5  7  9  1  5  1  3  7
 7  6  6  8  9  5  1  7  8  4  9  3  4  2  7  1
 1  8  4  2  4  2  8  6  6  7  6  2  5  8  8  3
 5  2  8  2  6  8  8  5  1  5  5  4  4  2  4  5
 7  5  4  1  6  2  7  2  6  3  3  8  9  8  1  4
 9  6  3  6  3  8  8  3  1  3  4  5  3  7  9  9
```

### num=x4 trans

```cpp
template <class TS, class TD>
__global__ void ldmatrix_x4_trans(TS *S, TD *D, int M, int N)
{
    __shared__ TS smem_src[256];
    for (int i = 0; i < M * N; ++i)
    {
        smem_src[i] = S[i];
    }
    __syncthreads();

    int tid = threadIdx.x;

    TS *smem_addr = smem_src + tid % M * N + tid / M * 8;

    uint32_t dst0, dst1, dst2, dst3;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_addr);
    asm volatile ("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));


    half_t r1, r2, r3, r4, r5, r6, r7, r8;
    r1.storage = static_cast<uint16_t>(dst0 & 0xFFFF);
    r2.storage = static_cast<uint16_t>(dst0 >> 16);
    r3.storage = static_cast<uint16_t>(dst1 & 0xFFFF);
    r4.storage = static_cast<uint16_t>(dst1 >> 16);
    r5.storage = static_cast<uint16_t>(dst2 & 0xFFFF);
    r6.storage = static_cast<uint16_t>(dst2 >> 16);
    r7.storage = static_cast<uint16_t>(dst3 & 0xFFFF);
    r8.storage = static_cast<uint16_t>(dst3 >> 16);

    int row = tid / 4;
    int col = tid % 4 * 2;
    D[row * N + col + 0] = r1;
    D[row * N + col + 1] = r2;
    D[(row + 8) * N + col + 0] = r3;
    D[(row + 8) * N + col + 1] = r4;
    D[row * N + col + 8] = r5;
    D[row * N + col + 9] = r6;
    D[(row + 8) * N + col + 8] = r7;
    D[(row + 8) * N + col + 9] = r8;
}
```
计算结果：
```cpp
m = 16, n = 16
The logical shape of A:
 1  3  1  4  5  4  8  9  6  5  7  7  2  3  1  6
 6  5  8  4  3  9  9  2  8  4  2  8  7  9  3  7
 2  1  9  4  4  7  3  7  9  1  5  8  3  3  4  8
 5  1  3  8  9  9  7  5  1  6  1  8  5  4  3  4
 2  9  8  6  4  1  1  2  1  3  9  2  3  4  7  8
 4  9  4  1  7  8  5  7  2  5  3  4  6  4  7  8
 3  5  2  5  4  9  6  2  3  5  3  3  8  1  1  9
 7  4  9  2  9  2  9  1  7  9  4  1  3  2  8  4
 4  7  8  5  7  4  7  7  6  9  9  3  7  1  2  3
 2  9  4  2  1  3  2  5  1  4  6  3  5  2  4  6
 7  9  2  4  3  8  1  7  5  7  9  1  5  1  3  7
 7  6  6  8  9  5  1  7  8  4  9  3  4  2  7  1
 1  8  4  2  4  2  8  6  6  7  6  2  5  8  8  3
 5  2  8  2  6  8  8  5  1  5  5  4  4  2  4  5
 7  5  4  1  6  2  7  2  6  3  3  8  9  8  1  4
 9  6  3  6  3  8  8  3  1  3  4  5  3  7  9  9

The logical shape of B:
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0

The copy result of B:
 1  6  2  5  2  4  3  7  6  8  9  1  1  2  3  7
 3  5  1  1  9  9  5  4  5  4  1  6  3  5  5  9
 1  8  9  3  8  4  2  9  7  2  5  1  9  3  3  4
 4  4  4  8  6  1  5  2  7  8  8  8  2  4  3  1
 5  3  4  9  4  7  4  9  2  7  3  5  3  6  8  3
 4  9  7  9  1  8  9  2  3  9  3  4  4  4  1  2
 8  9  3  7  1  5  6  9  1  3  4  3  7  7  1  8
 9  2  7  5  2  7  2  1  6  7  8  4  8  8  9  4
 4  2  7  7  1  5  7  9  6  1  5  8  6  1  6  1
 7  9  9  6  8  2  5  6  9  4  7  4  7  5  3  3
 8  4  2  6  4  8  4  3  9  6  9  9  6  5  3  4
 5  2  4  8  2  2  1  6  3  3  1  3  2  4  8  5
 7  1  3  9  4  6  6  3  7  5  5  4  5  4  9  3
 4  3  8  5  2  8  2  8  1  2  1  2  8  2  8  7
 7  2  1  1  8  8  7  8  2  4  3  7  8  4  1  9
 7  5  7  7  6  5  2  3  3  6  7  1  3  5  4  9
```

## stmatrix指令

### 指令介绍

stmatrix指令与ldmatrix类似，可以用于将寄存器中的数据保存到共享内存中。
使用方法也和ldmatrix类似，不过需要sm_90以上才可以使用。

```cpp
stmatrix.sync.aligned.shape.num{.trans}{.ss}.type [p], r;

.shape  = {.m8n8, .m16n8};
.num    = {.x1, .x2, .x4};
.ss     = {.shared{::cta}};
.type   = {.b16, .b8};
```

指令中.sync和.aligned的含义与ldmatrix中的相同，.sync代表stmatrix会使执行线程等待，直到 warp 中的所有线程都执行相同的 stmatrix 指令后才会继续执行。.aligned 限定符表示，warp 中的所有线程必须执行相同的 stmatrix 指令。
.shape代表一个指令加载的矩阵大小。stmatrix支持m8n8和m16n8两种形状，其中m16n8适用于bit8类型，m8n8适用于bit16类型。
.num可以是.x1，.x2和.x4，分别代表加载1个，2个和4个8*8大小的矩阵。

矩阵中每一行不需要在内存中连续存储。每个矩阵所需的八个地址由八个线程提供，具体取决于.num的值，如下表所示。每个地址对应矩阵行的起始位置。地址 `addr0` 到 `addr7` 对应第一个矩阵的行，地址 `addr8` 到 `addr15` 对应第二个矩阵的行，依此类推。


当num=x1时stmatrix保存第一个8*8矩阵中的64个16bit数据。只需要将8行中每行的首地址传给0-7号线程，则一个warp中32个线程的每一个线程会需要用一个32bit的寄存器保存2个16bit的数据，所以一共会保存64个数据。
当num=x2时stmatrix会保存2个8*8矩阵中的128个16bit数据。此时需要将16行中每行的首地址传给0-7和8-15号线程。一个warp中的32个线程中每一个线程都会使用2个32bit的寄存器保存4个16bit的数据，所以一共会保存两个矩阵中的128个数据。
当num=x4时stmatrix会保存4个8*8矩阵中的256个16bit数据。需要将32个行的首地址分别传给32个线程。一个线程需要使用4个32bit寄存器保存8个16bit的数据，所以可以保存4个8*8矩阵中的256个数据。

对于一个8*8矩阵，32个线程对应的保存数据如下图所示。每个线程在一个8*8矩阵中保存2个16bit数据，一行8个数据需要用4个线程保存，8行正好需要32个线程，这是num=x1的情况。
当num=x2时第二个矩阵的元素会按照上表中的布局保存到每个线程的第二个寄存器中。类似的当num=x4时，第三个和第四个矩阵的元素会保存到每个线程的后续的寄存器中。

可选的限定符.trans表示矩阵以列主序（column-major）格式保存。

代码测试
num=x4

```cpp
template <class TS, class TD>
__global__ void stmatrix_x4(TS *S, TD *D, int M, int N)
{
    int tid = threadIdx.x;
    __shared__ TS smem_src[256], smem_dst[256];
    for (int i = 0; i < 8; ++i)
    {
        smem_src[tid + i * 32] = S[tid + i * 32];
    }
    __syncthreads();

    TS *smem_src_addr = smem_src + tid % M * N + tid / M * 8;

    uint32_t dst0, dst1, dst2, dst3;
    uint32_t smem_src_ptr = cast_smem_ptr_to_uint(smem_src_addr);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
                 : "r"(smem_src_ptr));

    TS *smem_dst_addr = smem_dst + tid % M * N + tid / M * 8;
    uint32_t smem_dst_ptr = cast_smem_ptr_to_uint(smem_dst_addr);
    asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(smem_dst_ptr),
                 "r"(dst0), "r"(dst1), "r"(dst2), "r"(dst3));

    for (int i = 0; i < 8; ++i)
    {
        D[tid + i * 32] = smem_dst[tid + i * 32];
    }
}
```

测试结果
```cpp
m = 16, n = 16
The logical shape of A:
 1  3  1  4  5  4  8  9  6  5  7  7  2  3  1  6
 6  5  8  4  3  9  9  2  8  4  2  8  7  9  3  7
 2  1  9  4  4  7  3  7  9  1  5  8  3  3  4  8
 5  1  3  8  9  9  7  5  1  6  1  8  5  4  3  4
 2  9  8  6  4  1  1  2  1  3  9  2  3  4  7  8
 4  9  4  1  7  8  5  7  2  5  3  4  6  4  7  8
 3  5  2  5  4  9  6  2  3  5  3  3  8  1  1  9
 7  4  9  2  9  2  9  1  7  9  4  1  3  2  8  4
 4  7  8  5  7  4  7  7  6  9  9  3  7  1  2  3
 2  9  4  2  1  3  2  5  1  4  6  3  5  2  4  6
 7  9  2  4  3  8  1  7  5  7  9  1  5  1  3  7
 7  6  6  8  9  5  1  7  8  4  9  3  4  2  7  1
 1  8  4  2  4  2  8  6  6  7  6  2  5  8  8  3
 5  2  8  2  6  8  8  5  1  5  5  4  4  2  4  5
 7  5  4  1  6  2  7  2  6  3  3  8  9  8  1  4
 9  6  3  6  3  8  8  3  1  3  4  5  3  7  9  9

The copy result of B:
 1  3  1  4  5  4  8  9  6  5  7  7  2  3  1  6
 6  5  8  4  3  9  9  2  8  4  2  8  7  9  3  7
 2  1  9  4  4  7  3  7  9  1  5  8  3  3  4  8
 5  1  3  8  9  9  7  5  1  6  1  8  5  4  3  4
 2  9  8  6  4  1  1  2  1  3  9  2  3  4  7  8
 4  9  4  1  7  8  5  7  2  5  3  4  6  4  7  8
 3  5  2  5  4  9  6  2  3  5  3  3  8  1  1  9
 7  4  9  2  9  2  9  1  7  9  4  1  3  2  8  4
 4  7  8  5  7  4  7  7  6  9  9  3  7  1  2  3
 2  9  4  2  1  3  2  5  1  4  6  3  5  2  4  6
 7  9  2  4  3  8  1  7  5  7  9  1  5  1  3  7
 7  6  6  8  9  5  1  7  8  4  9  3  4  2  7  1
 1  8  4  2  4  2  8  6  6  7  6  2  5  8  8  3
 5  2  8  2  6  8  8  5  1  5  5  4  4  2  4  5
 7  5  4  1  6  2  7  2  6  3  3  8  9  8  1  4
 9  6  3  6  3  8  8  3  1  3  4  5  3  7  9  9
```

