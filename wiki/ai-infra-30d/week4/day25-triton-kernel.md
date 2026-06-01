---
title: "Day 25: Triton 和 Kernel 优化入门"
type: concept
tags: [day25, triton, cuda, kernel, gpu-programming]
created: 2026-06-01
updated: 2026-06-01
---

# Day 25: Triton 和 Kernel 优化入门

## Part 1: 为什么需要了解 GPU Kernel

推理优化的最底层就是 GPU kernel——每一次矩阵乘法、attention 计算、LayerNorm 都是一个或多个 kernel。

```
不需要成为 kernel 专家，但需要理解:
  1. kernel 是什么，怎么工作
  2. 为什么有些 kernel 快有些慢
  3. 如何判断 kernel 是 compute-bound 还是 memory-bound
  4. Triton 如何让写 kernel 更简单
```

---

## Part 2: CUDA 编程模型基础

### 线程层次

```
Grid (一个 kernel launch)
  └── Block 0, Block 1, ..., Block N    (在 SM 上执行)
        └── Thread 0, Thread 1, ..., Thread M  (一个 block 内)
              └── Warp (32 个连续 thread 组成)   (SIMD 执行单元)

GPU 执行单位:
  - Warp: 32 个 thread，锁步执行同一条指令
  - SM (Streaming Multiprocessor): 可以同时运行多个 warp
  - H100 有 132 个 SM
```

### 内存层次

```
寄存器 (Register):       每个 thread 私有，最快，容量小 (~256 KB/SM)
共享内存 (Shared Memory): 同一 block 内所有 thread 共享，快 (~228 KB/SM on H100)
L2 Cache:                全局共享，中等 (~50 MB)
全局内存 (HBM):           所有 thread 可访问，最慢但最大 (80 GB)

速度:  Register >> Shared Memory >> L2 >> HBM
       ~19 TB/s    ~19 TB/s       ~12 TB/s  ~3.35 TB/s
```

### 关键性能概念

**Memory Coalescing（合并访问）**
```
GPU 以 128 bytes 为单位从 HBM 读数据
如果一个 warp (32 threads) 访问连续的 32 × 4 bytes = 128 bytes → 1 次读取
如果每个 thread 访问分散的地址 → 可能需要 32 次读取

→ 连续内存访问比随机访问快 ~32x
```

**Occupancy（占用率）**
```
每个 SM 最多可同时运行的 warp 数 = max_warps_per_SM
实际在运行的 warp 数 / max_warps_per_SM = occupancy

高 occupancy: GPU 有更多 warp 来隐藏内存延迟
低 occupancy: 等待内存时 GPU 空闲
```

---

## Part 3: Triton 编程模型

### 为什么用 Triton

```
CUDA: 手动管理 thread、shared memory、sync → 复杂、容易出错、需要大量经验
Triton: 用 Python 写 "block-level" 程序 → 编译器自动管理底层细节

Triton 隐藏了:
  - thread 和 warp 的管理
  - shared memory 的分配和 bank conflict 避免
  - memory coalescing
  
Triton 不隐藏:
  - block 的形状和数量（你需要决定）
  - 算法逻辑
  - 数据加载和存储的模式
```

### Triton 核心概念

```python
import triton
import triton.language as tl

@triton.jit    # JIT 编译为 GPU 代码
def my_kernel(
    x_ptr,       # 输入指针
    y_ptr,       # 输出指针
    N,           # 数据大小
    BLOCK_SIZE: tl.constexpr,  # 编译时常量
):
    # 每个 "program instance" 处理一个 block
    pid = tl.program_id(0)    # 当前 block 的 ID
    
    # 计算这个 block 负责的数据范围
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # 加载数据（从 HBM 到寄存器）
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # 计算
    y = x * 2.0
    
    # 存储结果
    tl.store(y_ptr + offsets, y, mask=mask)

# 启动 kernel
grid = (triton.cdiv(N, BLOCK_SIZE),)
my_kernel[grid](x_ptr, y_ptr, N, BLOCK_SIZE=1024)
```

### Triton vs CUDA 对比

```
同一个 vector add kernel:

CUDA (C++):
  __global__ void add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
  }
  add<<<(n+255)/256, 256>>>(a, b, c, n);

Triton (Python):
  @triton.jit
  def add(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)
```

简单操作差别不大。但复杂操作（如 attention、matmul）中 Triton 的优势非常明显。

---

## Part 4: 实战——Triton Softmax

Softmax 是推理中非常常见的操作，也是理解 Triton 的好例子。

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program instance 处理一行
    row_idx = tl.program_id(0)
    
    # 计算行内偏移
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # 加载一行数据
    row_start_ptr = input_ptr + row_idx * input_row_stride
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Numerically stable softmax
    row_max = tl.max(row, axis=0)          # Step 1: max
    numerator = tl.exp(row - row_max)       # Step 2: exp(x - max)
    denominator = tl.sum(numerator, axis=0) # Step 3: sum
    softmax_output = numerator / denominator # Step 4: normalize
    
    # 存储结果
    output_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_start_ptr + col_offsets, softmax_output, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        output, x,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

# 使用
x = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
y_triton = softmax(x)
y_torch = torch.softmax(x, dim=-1)
assert torch.allclose(y_triton, y_torch, atol=1e-2)
```

### 这个 kernel 为什么高效

```
1. 每行只从 HBM 读一次（加载到寄存器），计算 max、exp、sum、normalize 全在寄存器内完成
2. 不需要将中间结果（如 exp 后的值）写回 HBM
3. 一个 kernel 完成 4 步操作（fusion）

vs PyTorch 标准实现可能需要多个 kernel:
  kernel 1: 找 max → 写回 HBM
  kernel 2: 计算 exp(x - max) → 写回 HBM
  kernel 3: 计算 sum → 写回 HBM
  kernel 4: 除以 sum → 写回 HBM
  
→ 4 次 HBM 读写 vs Triton 的 1 次
```

---

## Part 5: Kernel Fusion 在推理中的应用

### 常见 Fusion 模式

```
1. QKV Fusion:
   不分三次: Q = X@Wq, K = X@Wk, V = X@Wv
   而是一次: QKV = X @ [Wq; Wk; Wv]  → 1 个大 GEMM 替代 3 个小 GEMM

2. RMSNorm + 后续操作 Fusion:
   RMSNorm + QKV projection 融合
   减少一次 HBM 读写

3. SwiGLU Fusion:
   gate = X @ W_gate
   up = X @ W_up
   output = SiLU(gate) * up
   → gate 和 up 的 GEMM 合并，SiLU + element-wise multiply 在一个 kernel 中

4. Attention Fusion (FlashAttention):
   QK^T + softmax + V 的乘法在一个 kernel 中
   → 不写中间的 N×N attention matrix 到 HBM
```

### torch.compile 的自动 Fusion

```python
# PyTorch 2.0+ 的 torch.compile 可以自动发现并融合简单操作
@torch.compile
def fused_rmsnorm(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight

# torch.compile 会通过 TorchInductor 生成 Triton kernel
# 自动将 pow、mean、rsqrt、mul 融合为一个 kernel
```

---

## Part 6: Profiling Kernel

```bash
# 用 Nsight Compute 分析单个 kernel 的性能
ncu --set full python my_kernel_test.py

# 关键输出:
#   SM Throughput: xx%        ← SM 利用率
#   Memory Throughput: xx%    ← 内存带宽利用率
#   Arithmetic Intensity: xx  ← 计算/访存比
#   Achieved Occupancy: xx%   ← 占用率
```

---

## 交付物

| 文件 | 描述 |
|---|---|
| `triton-kernel-notes.md` | Triton softmax 实现 + kernel fusion 原理 + profiling 方法 |

## 自检问题

1. GPU 的 thread → warp → block → grid 层次各代表什么？
2. 什么是 memory coalescing？为什么它对性能重要？
3. Triton 和 CUDA 的编程模型有什么区别？
4. 为什么 fused softmax 比非 fused 版本快很多？
5. FlashAttention 本质上是一种 kernel fusion 吗？
