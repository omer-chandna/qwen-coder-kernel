#pragma once

#include <adakernel/api.h>

#include <cuda_runtime.h>

struct FatGemv {
    int active;
    int lock;
    int task_id;
    int next_tile;
    int tiles_left;
    int n_tiles;
    int tile_rows;
    int rows;
    int K;
    unsigned long long w;
    unsigned long long x;
    unsigned long long y;
};

struct ak_persist_buf {
    ak_task* d_tasks;
    int* d_succ;
    int* d_ready;
    int* d_flags;
    int* d_ready_init;
    int* d_cursor;
    int* d_reserved;
    int* d_completed;
    int* d_error;
    FatGemv* d_fat;
    int n_tasks;
    int n_edges;
    int n_ready;
    int n_blocks;
};

void ak_launch_gemv_grid(const float* W, const float* x, float* y, int rows, int K, int blocks, cudaStream_t stream);

ak_status ak_launch_task(const ak_task* task, cudaStream_t stream);
ak_status ak_persistent_run(const ak_task* tasks, int n_tasks, const int32_t* succ, int n_edges,
                            const int32_t* ready_init, int n_ready, int n_blocks, int max_idle,
                            cudaStream_t stream, int* out_completed);
int ak_persistent_occupancy(void);
int ak_persistent_regs(void);
ak_status ak_persist_alloc(ak_persist_buf* buf, int n_tasks, int n_edges, int n_ready, int n_blocks);
ak_status ak_persist_upload(ak_persist_buf* buf, const ak_task* tasks, const int32_t* succ, const int32_t* ready_init);
ak_status ak_persist_launch(ak_persist_buf* buf, int max_idle, cudaStream_t stream);
void ak_persist_free(ak_persist_buf* buf);
