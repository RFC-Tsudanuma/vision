// SPDX-FileCopyrightText: 2026 RFC-Tsudanuma
// SPDX-License-Identifier: Apache-2.0

#include "booster_vision/model/trt/gpu_preprocess.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <stdio.h>

namespace booster_vision {

namespace {

// Singleton pinned staging for source uint8 image. Reused across detectors.
uint8_t* g_host_staging = nullptr;
uint8_t* g_dev_staging = nullptr;
int g_max_bytes = 0;

__device__ inline float NormalizePixel(float v, float mean, float std_inv) {
    return (v * (1.0f / 255.0f) - mean) * std_inv;
}

__global__ void PreprocessKernel(
    const uint8_t* __restrict__ src, int src_stride, int src_w, int src_h,
    float* __restrict__ dst, int dst_w, int dst_h,
    float scale_inv, int dx, int dy,
    float pad_norm0, float pad_norm1, float pad_norm2,
    float mean0, float mean1, float mean2,
    float std0_inv, float std1_inv, float std2_inv,
    int bgr_to_rgb,
    int total_threads) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_threads) return;

    const int x = idx % dst_w;
    const int y = idx / dst_w;

    // Map destination pixel back into source space (nearest + clamped bilinear).
    const float fx = (x - dx) * scale_inv;
    const float fy = (y - dy) * scale_inv;

    float c0, c1, c2;
    if (fx < 0.0f || fy < 0.0f || fx >= src_w - 1.0f || fy >= src_h - 1.0f) {
        // Write normalized pad color directly.
        c0 = pad_norm0;
        c1 = pad_norm1;
        c2 = pad_norm2;
    } else {
        const int x_lo = (int)floorf(fx);
        const int y_lo = (int)floorf(fy);
        const int x_hi = x_lo + 1;
        const int y_hi = y_lo + 1;
        const float lx = fx - x_lo;
        const float ly = fy - y_lo;

        const uint8_t* p00 = src + y_lo * src_stride + x_lo * 3;
        const uint8_t* p10 = src + y_lo * src_stride + x_hi * 3;
        const uint8_t* p01 = src + y_hi * src_stride + x_lo * 3;
        const uint8_t* p11 = src + y_hi * src_stride + x_hi * 3;

        const float w00 = (1.0f - lx) * (1.0f - ly);
        const float w10 = lx * (1.0f - ly);
        const float w01 = (1.0f - lx) * ly;
        const float w11 = lx * ly;

        float b0 = w00 * p00[0] + w10 * p10[0] + w01 * p01[0] + w11 * p11[0];
        float b1 = w00 * p00[1] + w10 * p10[1] + w01 * p01[1] + w11 * p11[1];
        float b2 = w00 * p00[2] + w10 * p10[2] + w01 * p01[2] + w11 * p11[2];

        // Src is interpreted as BGR; if bgr_to_rgb, swap so channel 0 becomes R.
        float ch_r = bgr_to_rgb ? b2 : b0;
        float ch_g = b1;
        float ch_b = bgr_to_rgb ? b0 : b2;

        c0 = NormalizePixel(ch_r, mean0, std0_inv);
        c1 = NormalizePixel(ch_g, mean1, std1_inv);
        c2 = NormalizePixel(ch_b, mean2, std2_inv);
    }

    const int plane = dst_w * dst_h;
    dst[0 * plane + y * dst_w + x] = c0;
    dst[1 * plane + y * dst_w + x] = c1;
    dst[2 * plane + y * dst_w + x] = c2;
}

}  // namespace

void gpu_preprocess_init(int max_pixels) {
    const int bytes = max_pixels * 3;
    if (bytes <= g_max_bytes) return;

    gpu_preprocess_destroy();
    cudaMallocHost((void**)&g_host_staging, bytes);
    cudaMalloc((void**)&g_dev_staging, bytes);
    g_max_bytes = bytes;
}

void gpu_preprocess_destroy() {
    if (g_host_staging) {
        cudaFreeHost(g_host_staging);
        g_host_staging = nullptr;
    }
    if (g_dev_staging) {
        cudaFree(g_dev_staging);
        g_dev_staging = nullptr;
    }
    g_max_bytes = 0;
}

LetterboxInfo gpu_preprocess_compute_letterbox(int src_w, int src_h,
                                               int dst_w, int dst_h) {
    LetterboxInfo info;
    info.scale = std::min(static_cast<float>(dst_w) / src_w,
                          static_cast<float>(dst_h) / src_h);
    const int new_w = static_cast<int>(src_w * info.scale + 0.5f);
    const int new_h = static_cast<int>(src_h * info.scale + 0.5f);
    info.dx = (dst_w - new_w) / 2;
    info.dy = (dst_h - new_h) / 2;
    return info;
}

void gpu_preprocess_stage_host(const uint8_t* src_u8, int src_bytes) {
    if (src_bytes > g_max_bytes) {
        gpu_preprocess_init(src_bytes / 3 + 1);
    }
    memcpy(g_host_staging, src_u8, src_bytes);
}

void gpu_preprocess_run_gpu(int src_w, int src_h,
                            float* dst_f32_device, int dst_w, int dst_h,
                            const PreprocessParams& params,
                            cudaStream_t stream) {
    const int src_bytes = src_w * src_h * 3;
    cudaMemcpyAsync(g_dev_staging, g_host_staging, src_bytes,
                    cudaMemcpyHostToDevice, stream);

    const LetterboxInfo lb = gpu_preprocess_compute_letterbox(src_w, src_h, dst_w, dst_h);
    const float scale_inv = 1.0f / lb.scale;

    const float std0_inv = 1.0f / params.std[0];
    const float std1_inv = 1.0f / params.std[1];
    const float std2_inv = 1.0f / params.std[2];
    const float pad_f = params.pad_value * (1.0f / 255.0f);
    const float pad_n0 = (pad_f - params.mean[0]) * std0_inv;
    const float pad_n1 = (pad_f - params.mean[1]) * std1_inv;
    const float pad_n2 = (pad_f - params.mean[2]) * std2_inv;

    const int total = dst_w * dst_h;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    PreprocessKernel<<<blocks, threads, 0, stream>>>(
        g_dev_staging, src_w * 3, src_w, src_h,
        dst_f32_device, dst_w, dst_h,
        scale_inv, lb.dx, lb.dy,
        pad_n0, pad_n1, pad_n2,
        params.mean[0], params.mean[1], params.mean[2],
        std0_inv, std1_inv, std2_inv,
        params.bgr_to_rgb ? 1 : 0,
        total);
}

void gpu_preprocess_letterbox(const uint8_t* src_u8, int src_w, int src_h,
                              float* dst_f32_device, int dst_w, int dst_h,
                              const PreprocessParams& params,
                              cudaStream_t stream, LetterboxInfo* info) {
    const int src_bytes = src_w * src_h * 3;
    gpu_preprocess_stage_host(src_u8, src_bytes);
    gpu_preprocess_run_gpu(src_w, src_h, dst_f32_device, dst_w, dst_h,
                           params, stream);
    if (info) {
        *info = gpu_preprocess_compute_letterbox(src_w, src_h, dst_w, dst_h);
    }
}

}  // namespace booster_vision
