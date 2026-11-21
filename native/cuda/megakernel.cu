#include "error.h"
#include "launch.h"
#include "ops.cuh"

#include <cuda_runtime.h>

#include <cstring>

namespace {

__global__ void one_task_kernel(ak_task task) { akd::dispatch_task(task); }

__global__ void q8_grid_kernel(const uint8_t* W, const float* x, float* y, int rows, int K) {
    const int nwarps = blockDim.x >> 5;
    const int row0 = blockIdx.x * nwarps;
    if (row0 >= rows) {
        return;
    }
    int nrows = rows - row0;
    if (nrows > nwarps) {
        nrows = nwarps;
    }
    akd::q8_matvec_rows(W + static_cast<size_t>(row0) * ((K >> 5) * 34), x, y + row0, nrows, K);
}

__global__ void q6k_grid_kernel(const uint8_t* W, const float* x, float* y, int rows, int K) {
    const int nwarps = blockDim.x >> 5;
    const int row0 = blockIdx.x * nwarps;
    if (row0 >= rows) {
        return;
    }
    int nrows = rows - row0;
    if (nrows > nwarps) {
        nrows = nwarps;
    }
    akd::q6k_matvec_rows(W + static_cast<size_t>(row0) * ((K >> 8) * 210), x, y + row0, nrows, K);
}

// rows1 > 0: rows [rows1, rows) come from W2 (same type and K). The caller keeps rows1 a multiple of the
// per-block row count, so no block straddles the two matrices.
template <int T>
__global__ void dp4a_grid_kernel(const uint8_t* W, const uint8_t* W2, int rows1, const int8_t* xq, const float* xd,
                                 float* y, int rows, int K) {
    constexpr int bb = T == 0 ? 34 : (T == 1 ? 144 : 210);
    const int nb = T == 0 ? (K >> 5) : (K >> 8);
    const int per_block = (blockDim.x >> 5) * akd::qgemv_rows_per_warp(T);
    const int row0 = blockIdx.x * per_block;
    if (row0 >= rows) {
        return;
    }
    int nrows = rows - row0;
    if (nrows > per_block) {
        nrows = per_block;
    }
    const uint8_t* base = W;
    int wrow = row0;
    if (W2 && row0 >= rows1) {
        base = W2;
        wrow = row0 - rows1;
    }
    akd::qgemv_dp4a_rows<T>(base + static_cast<size_t>(wrow) * nb * bb, xq, xd, y + row0, nrows, K);
}

__global__ void qk_norm_rope_grid(ak_task t) {
    // 8 heads (one per warp) per block.
    const int h0 = blockIdx.x * (blockDim.x >> 5);
    if (h0 >= t.n0) return;
    const int nh = min(t.n0 - h0, static_cast<int>(blockDim.x >> 5));
    const int hd = t.n1;
    const int* pos = reinterpret_cast<const int*>(t.p3);
    akd::kv_t* kc = reinterpret_cast<akd::kv_t*>(t.p2);
    akd::kv_t* vc = reinterpret_cast<akd::kv_t*>(t.p5);
    // Heads are absolute indices, so the KV cache stays indexed [pos][n_heads][hd] with the full head count.
    akd::qk_norm_rope_range(reinterpret_cast<float*>(t.p0), reinterpret_cast<const float*>(t.p1), h0, nh, t.n0, hd,
                            t.f0, pos, __int_as_float(t.i0), kc, reinterpret_cast<const float*>(t.p4), vc, t.n2);
}

__global__ void attn_decode_grid(ak_task t) {
    // block = query head; i1 = n_rep
    const int head = blockIdx.x;
    const int hd = t.n1;
    akd::attn_decode_head(reinterpret_cast<const float*>(t.p0) + head * hd, reinterpret_cast<const float*>(t.p1),
                          reinterpret_cast<const float*>(t.p2), reinterpret_cast<int8_t*>(t.p3) + head * hd,
                          reinterpret_cast<float*>(t.p6) + head * (hd >> 5),
                          reinterpret_cast<float*>(t.p5) + static_cast<size_t>(head) * t.n2, head / t.i1, hd,
                          akd::attn_seq(t), t.n3);
}

__global__ void attn_part_grid(ak_task t) {
    akd::attn_part(reinterpret_cast<const float*>(t.p0), 0, reinterpret_cast<const akd::kv_t*>(t.p1),
                   reinterpret_cast<const akd::kv_t*>(t.p2), reinterpret_cast<float*>(t.p3), t.n0, t.n3, t.n1, t.n2, t.i0,
                   t.i1 & 0xFF, reinterpret_cast<const int*>(t.p4), blockIdx.x, blockIdx.y, 0);
}

__global__ void attn_combine_grid(ak_task t) {
    akd::attn_combine(reinterpret_cast<const float*>(t.p3), t.n0, t.n1, t.i1 & 0xFF, t.i0, t.n2,
                      reinterpret_cast<const int*>(t.p4), reinterpret_cast<int8_t*>(t.p5),
                      reinterpret_cast<float*>(t.p6), blockIdx.x, 0);
}

__global__ void argmax_grid(const float* v, int n, int base, unsigned long long* best) {
    const int span = 4096;
    const int lo = blockIdx.x * span;
    if (lo >= n) return;
    akd::argmax_part(v + lo, min(span, n - lo), base + lo, best);
}

__global__ void silu_quant_grid(const float* gate, const float* up, int8_t* xq, float* xd, int n) {
    const int lo = blockIdx.x * 256;
    if (lo >= n) return;
    akd::silu_quant(gate + lo, up + lo, xq + lo, xd + lo / 32, min(256, n - lo));
}

__global__ void q4k_grid_kernel(const uint8_t* W, const float* x, float* y, int rows, int K) {
    const int nwarps = blockDim.x >> 5;
    const int row0 = blockIdx.x * nwarps;
    if (row0 >= rows) {
        return;
    }
    int nrows = rows - row0;
    if (nrows > nwarps) {
        nrows = nwarps;
    }
    akd::q4k_matvec_rows(W + static_cast<size_t>(row0) * ((K >> 8) * 144), x, y + row0, nrows, K);
}

__global__ void gemv_grid_kernel(const float* W, const float* x, float* y, int rows, int K) {
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int global_warp = blockIdx.x * warps_per_block + warp_in_block;
    const int total_warps = gridDim.x * warps_per_block;
    const bool vec = ((reinterpret_cast<uintptr_t>(W) | reinterpret_cast<uintptr_t>(x)) & 0xF) == 0 && (K & 3) == 0;
    for (int row = global_warp; row < rows; row += total_warps) {
        const float* wrow = W + static_cast<size_t>(row) * static_cast<size_t>(K);
        float acc = 0.f;
        if (vec) {
            const float4* w4 = reinterpret_cast<const float4*>(wrow);
            const float4* x4 = reinterpret_cast<const float4*>(x);
            const int k4 = K >> 2;
            for (int j = lane; j < k4; j += 32) {
                const float4 a = w4[j];
                const float4 b = x4[j];
                acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        } else {
            for (int j = lane; j < K; j += 32) {
                acc += wrow[j] * x[j];
            }
        }
        acc = akd::warp_sum(acc);
        if (lane == 0) {
            y[row] = acc;
        }
    }
}

__device__ void finish_task(const ak_task& task, ak_task* tasks, const int* succ, int* ready, int* flags,
                           int* reserved, int* completed) {
    for (int i = 0; i < task.n_succ; ++i) {
        const int child = succ[task.succ_begin + i];
        const int previous = atomicSub(&tasks[child].preds_left, 1);
        if (previous == 1) {
            const int pos = atomicAdd(reserved, 1);
            ready[pos] = child;
            __threadfence();
            flags[pos] = 1;
        }
    }
    atomicAdd(completed, 1);
}

__device__ void help_one_fat_tile(FatGemv* fat, ak_task* tasks, const int* succ, int* ready, int* flags, int* reserved,
                                  int* completed) {
    __shared__ int go;
    __shared__ int tile;
    __shared__ int n_tiles;
    __shared__ int tile_rows;
    __shared__ int rows;
    __shared__ int K;
    __shared__ int task_id;
    __shared__ unsigned long long wbits;
    __shared__ unsigned long long xbits;
    __shared__ unsigned long long ybits;
    if (threadIdx.x == 0) {
        go = *reinterpret_cast<volatile int*>(&fat->active);
        tile = -1;
        n_tiles = 0;
        if (go) {
            __threadfence();
            n_tiles = fat->n_tiles;
            tile_rows = fat->tile_rows;
            rows = fat->rows;
            K = fat->K;
            task_id = fat->task_id;
            wbits = fat->w;
            xbits = fat->x;
            ybits = fat->y;
            tile = atomicAdd(&fat->next_tile, 1);
        }
    }
    __syncthreads();
    if (!go || tile < 0 || tile >= n_tiles) {
        __syncthreads();
        return;
    }
    const int row0 = tile * tile_rows;
    int nrows = rows - row0;
    if (nrows > tile_rows) {
        nrows = tile_rows;
    }
    const float* base = reinterpret_cast<const float*>(wbits);
    float* out = reinterpret_cast<float*>(ybits);
    akd::matvec_rows(base + static_cast<size_t>(row0) * K, reinterpret_cast<const float*>(xbits), out + row0, nrows, K);
    if (threadIdx.x == 0) {
        const int left = atomicSub(&fat->tiles_left, 1);
        if (left == 1) {
            finish_task(tasks[task_id], tasks, succ, ready, flags, reserved, completed);
            __threadfence();
            fat->active = 0;
        }
    }
    __syncthreads();
}

__global__ void reset_kernel(ak_task* tasks, int n_tasks, const int* ready_init, int n_ready, int* ready, int* flags,
                             int* cursor, int* reserved, int* completed, int* error, FatGemv* fat) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_tasks; i += stride) {
        tasks[i].preds_left = tasks[i].preds_init;
        if (i < n_ready) {
            ready[i] = ready_init[i];
            flags[i] = 1;
        } else {
            flags[i] = 0;
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *cursor = 0;
        *reserved = n_ready;
        *completed = 0;
        *error = 0;
        if (fat) {
            fat->active = 0;
            fat->lock = 0;
            fat->next_tile = 0;
            fat->tiles_left = 0;
            fat->n_tiles = 0;
        }
    }
}

__global__ void __launch_bounds__(akd::kThreads, 3) persist_kernel(ak_task* tasks, const int* succ, int n_tasks, int* ready, int* flags, int* cursor,
                               int* reserved, int* completed, int* error, FatGemv* fat, int max_cycles) {
    __shared__ int sid;
    __shared__ int status;
    __shared__ int ticket_held;
    __shared__ int flag;
    __shared__ int fat_on;
    __shared__ int err;
    __shared__ int timed_out;
    __shared__ int won;
    const unsigned long long budget = max_cycles > 0 ? static_cast<unsigned long long>(max_cycles) : 1000000000ull;
    while (true) {
        if (threadIdx.x == 0) {
            status = 2;
            sid = -1;
            err = *reinterpret_cast<volatile int*>(error);
            const int done = *reinterpret_cast<volatile int*>(completed);
            fat_on = fat ? *reinterpret_cast<volatile int*>(&fat->active) : 0;
            if (err != 0 || (done >= n_tasks && !fat_on)) {
                status = 2;
            } else if (fat_on) {
                status = 1;
            } else {
                const int ticket = atomicAdd(cursor, 1);
                if (ticket >= n_tasks) {
                    status = 2;
                } else {
                    ticket_held = ticket;
                    status = 0;
                }
            }
        }
        __syncthreads();
        if (status == 2) {
            break;
        }
        if (status == 1) {
            help_one_fat_tile(fat, tasks, succ, ready, flags, reserved, completed);
            continue;
        }

        const unsigned long long start = clock64();
        while (true) {
            if (threadIdx.x == 0) {
                flag = reinterpret_cast<volatile int*>(flags)[ticket_held];
                fat_on = fat ? *reinterpret_cast<volatile int*>(&fat->active) : 0;
                err = *reinterpret_cast<volatile int*>(error);
                timed_out = (clock64() - start) > budget;
                if (timed_out) {
                    atomicExch(error, AK_ERR_TIMEOUT);
                }
            }
            __syncthreads();
            if (err != 0 || timed_out) {
                status = 2;
                break;
            }
            if (flag != 0) {
                break;
            }
            if (fat_on) {
                help_one_fat_tile(fat, tasks, succ, ready, flags, reserved, completed);
            }
        }
        __syncthreads();
        if (status == 2) {
            break;
        }
        if (threadIdx.x == 0) {
            __threadfence();
            sid = ready[ticket_held];
            if (sid < 0 || sid >= n_tasks) {
                atomicExch(error, AK_ERR_INVALID);
                sid = -1;
            }
        }
        __syncthreads();
        if (sid < 0) {
            break;
        }
        const ak_task claimed = tasks[sid];
        if (claimed.op == AK_OP_MATVEC && claimed.i0 == 1 && fat) {
            while (true) {
                if (threadIdx.x == 0) {
                    won = 0;
                    if (atomicCAS(&fat->lock, 0, 1) == 0) {
                        const int tile_rows = 32;
                        int n_tiles = (claimed.n0 + tile_rows - 1) / tile_rows;
                        if (n_tiles < 1) {
                            n_tiles = 1;
                        }
                        fat->task_id = sid;
                        fat->rows = claimed.n0;
                        fat->K = claimed.n1;
                        fat->tile_rows = tile_rows;
                        fat->n_tiles = n_tiles;
                        fat->tiles_left = n_tiles;
                        fat->next_tile = 0;
                        fat->w = claimed.p0;
                        fat->x = claimed.p1;
                        fat->y = claimed.p2;
                        __threadfence();
                        fat->active = 1;
                        atomicExch(&fat->lock, 0);
                        won = 1;
                    }
                }
                __syncthreads();
                if (won) {
                    break;
                }
                help_one_fat_tile(fat, tasks, succ, ready, flags, reserved, completed);
            }
            continue;
        }
        __threadfence();
        akd::dispatch_task(claimed);
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            finish_task(claimed, tasks, succ, ready, flags, reserved, completed);
        }
        __syncthreads();
    }
}

ak_status check_launch(cudaError_t err) {
    if (err != cudaSuccess) {
        ak_set_error(cudaGetErrorString(err));
        return AK_ERR_CUDA;
    }
    return AK_OK;
}

}  // namespace

void ak_launch_gemv_grid(const float* W, const float* x, float* y, int rows, int K, int blocks, cudaStream_t stream) {
    if (blocks < 1) {
        blocks = 1;
    }
    gemv_grid_kernel<<<blocks, akd::kThreads, 0, stream>>>(W, x, y, rows, K);
}

ak_status ak_launch_task(const ak_task* task, cudaStream_t stream) {
    if (!task) {
        ak_set_error("null task");
        return AK_ERR_INVALID;
    }
    if (task->op == AK_OP_MATVEC && task->i0 == 1) {
        int blocks = ak_persistent_occupancy();
        ak_launch_gemv_grid(reinterpret_cast<const float*>(task->p0), reinterpret_cast<const float*>(task->p1),
                            reinterpret_cast<float*>(task->p2), task->n0, task->n1, blocks, stream);
        return check_launch(cudaGetLastError());
    }
    if (task->op == AK_OP_QK_NORM_ROPE) {
        qk_norm_rope_grid<<<(task->n0 + 7) / 8, akd::kThreads, 0, stream>>>(*task);
        return check_launch(cudaGetLastError());
    }
    if (task->op == AK_OP_ATTN_PART && task->f0 > 0.5f) {
        attn_part_grid<<<dim3(task->n3, task->i1 & 0xFF), akd::kThreads, 0, stream>>>(*task);
        return check_launch(cudaGetLastError());
    }
    if (task->op == AK_OP_ATTN_COMBINE && task->f0 > 0.5f) {
        attn_combine_grid<<<task->n0, 128, 0, stream>>>(*task);
        return check_launch(cudaGetLastError());
    }
    if (task->op == AK_OP_ATTN_DECODE && task->i0 > 0) {
        attn_decode_grid<<<task->i0, akd::kThreads, 0, stream>>>(*task);
        return check_launch(cudaGetLastError());
    }
    if (task->op == AK_OP_ARGMAX_PART) {
        argmax_grid<<<(task->n0 + 4095) / 4096, akd::kThreads, 0, stream>>>(
            reinterpret_cast<const float*>(task->p0), task->n0, task->i0,
            reinterpret_cast<unsigned long long*>(task->p1));
        return check_launch(cudaGetLastError());
    }
    if (task->op == AK_OP_SILU_QUANT) {
        silu_quant_grid<<<(task->n0 + 255) / 256, akd::kThreads, 0, stream>>>(
            reinterpret_cast<const float*>(task->p0), reinterpret_cast<const float*>(task->p1),
            reinterpret_cast<int8_t*>(task->p2), reinterpret_cast<float*>(task->p3), task->n0);
        return check_launch(cudaGetLastError());
    }
    if (task->op == AK_OP_QUANT_Q8_1) {
        akd::quantize_q8_1_grid<<<(task->n0 / 32 + 7) / 8, 256, 0, stream>>>(
            reinterpret_cast<const float*>(task->p0), reinterpret_cast<int8_t*>(task->p1),
            reinterpret_cast<float*>(task->p3), task->n0);
        return check_launch(cudaGetLastError());
    }
    if (task->op == AK_OP_Q8_DP4A || task->op == AK_OP_Q4K_DP4A || task->op == AK_OP_Q6K_DP4A) {
        // 128-thread blocks (4 rows each) measured best on the 4060; 72 registers on the Q4_K path.
        const int threads = 128;
        const int type = task->op == AK_OP_Q8_DP4A ? 0 : (task->op == AK_OP_Q4K_DP4A ? 1 : 2);
        const int per_block = (threads >> 5) * akd::qgemv_rows_per_warp(type);
        const int blocks = (task->n0 + per_block - 1) / per_block;
        const uint8_t* W = reinterpret_cast<const uint8_t*>(task->p0);
        const int8_t* xq = reinterpret_cast<const int8_t*>(task->p1);
        const float* xd = reinterpret_cast<const float*>(task->p3);
        float* y = reinterpret_cast<float*>(task->p2);
        const uint8_t* W2 = reinterpret_cast<const uint8_t*>(task->p4);
        const int rows1 = W2 ? task->n2 : 0;
        if (W2 && (rows1 % per_block) != 0) {
            ak_set_error("dp4a pair: first matrix rows must be a multiple of the rows per block");
            return AK_ERR_INVALID;
        }
        if (task->op == AK_OP_Q8_DP4A) {
            dp4a_grid_kernel<0><<<blocks, threads, 0, stream>>>(W, W2, rows1, xq, xd, y, task->n0, task->n1);
        } else if (task->op == AK_OP_Q4K_DP4A) {
            dp4a_grid_kernel<1><<<blocks, threads, 0, stream>>>(W, W2, rows1, xq, xd, y, task->n0, task->n1);
        } else {
            dp4a_grid_kernel<2><<<blocks, threads, 0, stream>>>(W, W2, rows1, xq, xd, y, task->n0, task->n1);
        }
        return check_launch(cudaGetLastError());
    }
    if (task->op == AK_OP_Q8_MATVEC || task->op == AK_OP_Q4K_MATVEC || task->op == AK_OP_Q6K_MATVEC) {
        const int nwarps = akd::kThreads >> 5;
        int blocks = (task->n0 + nwarps - 1) / nwarps;
        if (blocks < 1) {
            blocks = 1;
        }
        if (task->op == AK_OP_Q8_MATVEC) {
            q8_grid_kernel<<<blocks, akd::kThreads, 0, stream>>>(reinterpret_cast<const uint8_t*>(task->p0),
                                                                reinterpret_cast<const float*>(task->p1),
                                                                reinterpret_cast<float*>(task->p2), task->n0, task->n1);
        } else if (task->op == AK_OP_Q6K_MATVEC) {
            q6k_grid_kernel<<<blocks, akd::kThreads, 0, stream>>>(reinterpret_cast<const uint8_t*>(task->p0),
                                                                 reinterpret_cast<const float*>(task->p1),
                                                                 reinterpret_cast<float*>(task->p2), task->n0, task->n1);
        } else {
            q4k_grid_kernel<<<blocks, akd::kThreads, 0, stream>>>(reinterpret_cast<const uint8_t*>(task->p0),
                                                                 reinterpret_cast<const float*>(task->p1),
                                                                 reinterpret_cast<float*>(task->p2), task->n0, task->n1);
        }
        return check_launch(cudaGetLastError());
    }
    one_task_kernel<<<1, akd::kThreads, 0, stream>>>(*task);
    return check_launch(cudaGetLastError());
}

int ak_persistent_occupancy(void) {
    int blocks = 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return 1;
    }
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, persist_kernel, akd::kThreads, 0);
    if (err != cudaSuccess || blocks < 1) {
        return prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
    }
    const int resident = blocks * prop.multiProcessorCount;
    return resident > 0 ? resident : 1;
}

ak_status ak_persistent_run(const ak_task* tasks, int n_tasks, const int32_t* succ, int n_edges,
                            const int32_t* ready_init, int n_ready, int n_blocks, int max_idle,
                            cudaStream_t stream, int* out_completed) {
    if (n_tasks < 0 || n_edges < 0 || n_ready < 0 || (n_tasks > 0 && !tasks) || (n_edges > 0 && !succ) ||
        (n_ready > 0 && !ready_init)) {
        ak_set_error("persistent_run: invalid arguments");
        return AK_ERR_INVALID;
    }
    if (n_tasks == 0) {
        if (out_completed) {
            *out_completed = 0;
        }
        return AK_OK;
    }
    const int cap = ak_persistent_occupancy();
    if (n_blocks <= 0 || n_blocks > cap) {
        n_blocks = cap;
    }
    ak_task* d_tasks = nullptr;
    int* d_succ = nullptr;
    int* d_ready = nullptr;
    int* d_ready_init = nullptr;
    int* d_flags = nullptr;
    int* d_cursor = nullptr;
    int* d_reserved = nullptr;
    int* d_completed = nullptr;
    int* d_error = nullptr;
    FatGemv* d_fat = nullptr;
    ak_status st = AK_OK;
    auto fail = [&](ak_status code) {
        cudaFree(d_tasks);
        cudaFree(d_succ);
        cudaFree(d_ready);
        cudaFree(d_ready_init);
        cudaFree(d_flags);
        cudaFree(d_cursor);
        cudaFree(d_reserved);
        cudaFree(d_completed);
        cudaFree(d_error);
        cudaFree(d_fat);
        return code;
    };
    if (cudaMalloc(&d_tasks, sizeof(ak_task) * n_tasks) != cudaSuccess ||
        cudaMalloc(&d_ready, sizeof(int) * n_tasks) != cudaSuccess ||
        cudaMalloc(&d_flags, sizeof(int) * n_tasks) != cudaSuccess ||
        cudaMalloc(&d_cursor, sizeof(int)) != cudaSuccess || cudaMalloc(&d_reserved, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_completed, sizeof(int)) != cudaSuccess || cudaMalloc(&d_error, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_fat, sizeof(FatGemv)) != cudaSuccess) {
        ak_set_error("persistent_run: out of memory");
        return fail(AK_ERR_OOM);
    }
    if (n_edges > 0 && cudaMalloc(&d_succ, sizeof(int) * n_edges) != cudaSuccess) {
        ak_set_error("persistent_run: out of memory");
        return fail(AK_ERR_OOM);
    }
    if (n_ready > 0 && cudaMalloc(&d_ready_init, sizeof(int) * n_ready) != cudaSuccess) {
        ak_set_error("persistent_run: out of memory");
        return fail(AK_ERR_OOM);
    }
    if (cudaMemcpyAsync(d_tasks, tasks, sizeof(ak_task) * n_tasks, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        (n_edges > 0 && cudaMemcpyAsync(d_succ, succ, sizeof(int) * n_edges, cudaMemcpyHostToDevice, stream) != cudaSuccess) ||
        (n_ready > 0 &&
         cudaMemcpyAsync(d_ready_init, ready_init, sizeof(int) * n_ready, cudaMemcpyHostToDevice, stream) != cudaSuccess)) {
        ak_set_error(cudaGetErrorString(cudaGetLastError()));
        return fail(AK_ERR_CUDA);
    }
    reset_kernel<<<1, 256, 0, stream>>>(d_tasks, n_tasks, d_ready_init, n_ready, d_ready, d_flags, d_cursor, d_reserved,
                                       d_completed, d_error, d_fat);
    persist_kernel<<<n_blocks, akd::kThreads, 0, stream>>>(d_tasks, d_succ, n_tasks, d_ready, d_flags, d_cursor,
                                                          d_reserved, d_completed, d_error, d_fat, max_idle);
    int completed = 0;
    int error = 0;
    if (cudaMemcpyAsync(&completed, d_completed, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(&error, d_error, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        ak_set_error(cudaGetErrorString(cudaGetLastError()));
        return fail(AK_ERR_CUDA);
    }
    if (out_completed) {
        *out_completed = completed;
    }
    cudaFree(d_tasks);
    cudaFree(d_succ);
    cudaFree(d_ready);
    cudaFree(d_ready_init);
    cudaFree(d_flags);
    cudaFree(d_cursor);
    cudaFree(d_reserved);
    cudaFree(d_completed);
    cudaFree(d_error);
    cudaFree(d_fat);
    if (error != 0) {
        ak_set_error(error == AK_ERR_TIMEOUT ? "persistent kernel timed out" : "persistent kernel failed");
        return static_cast<ak_status>(error);
    }
    if (completed != n_tasks) {
        ak_set_error("persistent kernel exited before every task finished");
        return AK_ERR_TIMEOUT;
    }
    (void)st;
    return AK_OK;
}

int ak_persistent_regs(void) {
    cudaFuncAttributes attr{};
    if (cudaFuncGetAttributes(&attr, persist_kernel) != cudaSuccess) {
        return 0;
    }
    return attr.numRegs;
}

ak_status ak_persist_alloc(ak_persist_buf* buf, int n_tasks, int n_edges, int n_ready, int n_blocks) {
    if (!buf || n_tasks <= 0) {
        ak_set_error("persist alloc: bad args");
        return AK_ERR_INVALID;
    }
    std::memset(buf, 0, sizeof(*buf));
    buf->n_tasks = n_tasks;
    buf->n_edges = n_edges;
    buf->n_ready = n_ready;
    const int cap = ak_persistent_occupancy();
    buf->n_blocks = (n_blocks <= 0 || n_blocks > cap) ? cap : n_blocks;
    cudaError_t err = cudaSuccess;
    auto bomb = [&](cudaError_t e) {
        if (err == cudaSuccess && e != cudaSuccess) {
            err = e;
        }
    };
    bomb(cudaMalloc(&buf->d_tasks, sizeof(ak_task) * n_tasks));
    bomb(cudaMalloc(&buf->d_ready, sizeof(int) * n_tasks));
    bomb(cudaMalloc(&buf->d_flags, sizeof(int) * n_tasks));
    bomb(cudaMalloc(&buf->d_cursor, sizeof(int)));
    bomb(cudaMalloc(&buf->d_reserved, sizeof(int)));
    bomb(cudaMalloc(&buf->d_completed, sizeof(int)));
    bomb(cudaMalloc(&buf->d_error, sizeof(int)));
    bomb(cudaMalloc(&buf->d_fat, sizeof(FatGemv)));
    if (n_edges > 0) {
        bomb(cudaMalloc(&buf->d_succ, sizeof(int) * n_edges));
    }
    if (n_ready > 0) {
        bomb(cudaMalloc(&buf->d_ready_init, sizeof(int) * n_ready));
    }
    if (err != cudaSuccess) {
        ak_set_error(cudaGetErrorString(err));
        ak_persist_free(buf);
        return err == cudaErrorMemoryAllocation ? AK_ERR_OOM : AK_ERR_CUDA;
    }
    return AK_OK;
}

ak_status ak_persist_upload(ak_persist_buf* buf, const ak_task* tasks, const int32_t* succ, const int32_t* ready_init) {
    if (!buf || !buf->d_tasks || !tasks) {
        ak_set_error("persist upload: bad args");
        return AK_ERR_INVALID;
    }
    if (cudaMemcpy(buf->d_tasks, tasks, sizeof(ak_task) * buf->n_tasks, cudaMemcpyHostToDevice) != cudaSuccess) {
        ak_set_error(cudaGetErrorString(cudaGetLastError()));
        return AK_ERR_CUDA;
    }
    if (buf->n_edges > 0 &&
        cudaMemcpy(buf->d_succ, succ, sizeof(int) * buf->n_edges, cudaMemcpyHostToDevice) != cudaSuccess) {
        ak_set_error(cudaGetErrorString(cudaGetLastError()));
        return AK_ERR_CUDA;
    }
    if (buf->n_ready > 0 &&
        cudaMemcpy(buf->d_ready_init, ready_init, sizeof(int) * buf->n_ready, cudaMemcpyHostToDevice) != cudaSuccess) {
        ak_set_error(cudaGetErrorString(cudaGetLastError()));
        return AK_ERR_CUDA;
    }
    return AK_OK;
}

ak_status ak_persist_launch(ak_persist_buf* buf, int max_idle, cudaStream_t stream) {
    if (!buf || !buf->d_tasks) {
        ak_set_error("persist launch: not allocated");
        return AK_ERR_INVALID;
    }
    reset_kernel<<<1, 256, 0, stream>>>(buf->d_tasks, buf->n_tasks, buf->d_ready_init, buf->n_ready, buf->d_ready,
                                       buf->d_flags, buf->d_cursor, buf->d_reserved, buf->d_completed, buf->d_error,
                                       buf->d_fat);
    persist_kernel<<<buf->n_blocks, akd::kThreads, 0, stream>>>(buf->d_tasks, buf->d_succ, buf->n_tasks, buf->d_ready,
                                                               buf->d_flags, buf->d_cursor, buf->d_reserved,
                                                               buf->d_completed, buf->d_error, buf->d_fat, max_idle);
    return check_launch(cudaGetLastError());
}

void ak_persist_free(ak_persist_buf* buf) {
    if (!buf) {
        return;
    }
    cudaFree(buf->d_tasks);
    cudaFree(buf->d_succ);
    cudaFree(buf->d_ready);
    cudaFree(buf->d_flags);
    cudaFree(buf->d_ready_init);
    cudaFree(buf->d_cursor);
    cudaFree(buf->d_reserved);
    cudaFree(buf->d_completed);
    cudaFree(buf->d_error);
    cudaFree(buf->d_fat);
    std::memset(buf, 0, sizeof(*buf));
}

extern "C" ak_status ak_persistent_execute(const ak_task* tasks, int n_tasks, const int32_t* succ, int n_edges,
                                           int n_blocks, int max_idle) {
    if (n_tasks < 0) {
        ak_set_error("negative task count");
        return AK_ERR_INVALID;
    }
    int ready_stack[4096];
    int n_ready = 0;
    if (n_tasks > 4096) {
        ak_set_error("persistent_execute supports at most 4096 tasks");
        return AK_ERR_INVALID;
    }
    for (int i = 0; i < n_tasks; ++i) {
        if (tasks[i].preds_init == 0) {
            ready_stack[n_ready++] = i;
        }
    }
    return ak_persistent_run(tasks, n_tasks, succ, n_edges, ready_stack, n_ready, n_blocks, max_idle, 0, nullptr);
}
