#include "error.h"
#include "launch.h"

#include <adakernel/api.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace {
int g_ready = 0;
}

extern "C" ak_status ak_init(int device) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        ak_set_error(err == cudaSuccess ? "no CUDA device" : cudaGetErrorString(err));
        return AK_ERR_CUDA;
    }
    if (device < 0 || device >= count) {
        ak_set_error("device index out of range");
        return AK_ERR_INVALID;
    }
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        ak_set_error(cudaGetErrorString(err));
        return AK_ERR_CUDA;
    }
    // Mapped pinned memory is optional. Flags must be set before the primary context exists.
    if (cudaSetDeviceFlags(cudaDeviceMapHost) != cudaSuccess) {
        cudaGetLastError();
    }
    err = cudaFree(nullptr);
    if (err != cudaSuccess) {
        ak_set_error(cudaGetErrorString(err));
        return AK_ERR_CUDA;
    }
    g_ready = 1;
    return AK_OK;
}

extern "C" ak_status ak_sync(void) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        ak_set_error(cudaGetErrorString(err));
        return AK_ERR_CUDA;
    }
    return AK_OK;
}

extern "C" ak_status ak_device_query(ak_device_info* out) {
    if (!out) {
        ak_set_error("null device info");
        return AK_ERR_INVALID;
    }
    std::memset(out, 0, sizeof(*out));
    if (!g_ready) {
        ak_status st = ak_init(0);
        if (st != AK_OK) {
            return st;
        }
    }
    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        ak_set_error(cudaGetErrorString(err));
        return AK_ERR_CUDA;
    }
    std::snprintf(out->name, sizeof(out->name), "%s", prop.name);
    cudaDeviceGetPCIBusId(out->pci_bus_id, sizeof(out->pci_bus_id), 0);
    out->cc_major = prop.major;
    out->cc_minor = prop.minor;
    out->sm_count = prop.multiProcessorCount;
    out->warp_size = prop.warpSize;
    out->max_threads_per_block = prop.maxThreadsPerBlock;
    out->max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    out->regs_per_sm = prop.regsPerMultiprocessor;
    out->shared_mem_per_sm = static_cast<int>(prop.sharedMemPerMultiprocessor);
    out->shared_mem_per_block = static_cast<int>(prop.sharedMemPerBlock);
    out->shared_mem_per_block_optin = static_cast<int>(prop.sharedMemPerBlockOptin);
    out->l2_cache_bytes = static_cast<int>(prop.l2CacheSize);
    out->pci_bus = prop.pciBusID;
    out->pci_device = prop.pciDeviceID;
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, 0);
    if (clock_khz <= 0) {
        clock_khz = prop.memoryClockRate;
    }
    out->memory_clock_khz = clock_khz;
    out->memory_bus_width_bits = prop.memoryBusWidth;
    int sm_clock = 0;
    cudaDeviceGetAttribute(&sm_clock, cudaDevAttrClockRate, 0);
    out->clock_sm_khz = sm_clock > 0 ? sm_clock : prop.clockRate;
    int coop = 0;
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0);
    out->cooperative_launch = coop;
    size_t free_b = 0;
    size_t total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    out->total_vram_mb = static_cast<float>(total_b / (1024.0 * 1024.0));
    out->free_vram_mb = static_cast<float>(free_b / (1024.0 * 1024.0));
    if (clock_khz > 0 && prop.memoryBusWidth > 0) {
        const double bytes_per_s = 2.0 * (static_cast<double>(clock_khz) * 1000.0) * (prop.memoryBusWidth / 8.0);
        out->theoretical_bandwidth_gbps = static_cast<float>(bytes_per_s / 1.0e9);
    }
    out->persistent_max_blocks = ak_persistent_occupancy();
    return AK_OK;
}
