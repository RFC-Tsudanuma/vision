// SPDX-FileCopyrightText: 2026 RFC-Tsudanuma
// SPDX-License-Identifier: Apache-2.0
//
// Parameterized GPU preprocess for TensorRT object detectors.
//
// Pipeline (single kernel, zero host-side math on the pixels):
//   BGR uint8  -> letterbox (aspect-preserving resize)
//              -> BGR/RGB swap (configurable)
//              -> per-channel normalize (pixel/255 - mean) / std
//              -> planar CHW float32 into caller-owned device buffer
//
// Used by RFDetrDetectorTRT. The parameters also cover YOLO-style detectors
// if that path is moved to the shared GPU preprocessor later.
//
// For YOLOv8-style detectors:   mean=(0,0,0), std=(1,1,1), pad=114, bgr_to_rgb=true
// For RF-DETR (DETR, ImageNet): mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
//                               pad=0, bgr_to_rgb=true
//
// The returned LetterboxInfo lets the caller unproject model-space bboxes
// back to source-image pixels in its own postprocess.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace booster_vision {

struct LetterboxInfo {
    float scale;  // model_size / max(src_w, src_h) style factor applied to both axes
    int dx;       // left padding in model-space pixels
    int dy;       // top padding in model-space pixels
};

struct PreprocessParams {
    float mean[3];      // RGB order (applied AFTER bgr_to_rgb if enabled)
    float std[3];       // RGB order
    uint8_t pad_value;  // fill value for letterbox padding, pre-normalization
    bool bgr_to_rgb;    // swap channel order (MuJoCo publishes rgb8 but cv::imread gives bgr)
};

// Call once at process start to allocate pinned host + device staging buffers
// sized for `max_pixels` source pixels of RGB/BGR input. Safe to call multiple
// times; only first allocation takes effect.
void gpu_preprocess_init(int max_pixels);
void gpu_preprocess_destroy();

// Copies `src_u8` (HWC uint8, `src_w * src_h * 3` bytes) to the device, runs
// the preprocess kernel, and writes planar CHW float32 into `dst_f32_device`
// (caller-allocated, `dst_w * dst_h * 3 * 4` bytes). Fills `info` with the
// letterbox parameters so postprocess can map back to source coords.
// `stream` is used for the memcpy and kernel launch.
void gpu_preprocess_letterbox(const uint8_t* src_u8, int src_w, int src_h,
                              float* dst_f32_device, int dst_w, int dst_h,
                              const PreprocessParams& params,
                              cudaStream_t stream, LetterboxInfo* info);

// ── CUDA Graph split API ────────────────────────────────────────────────
// The single-call API above does a host-side memcpy (src -> pinned) and then
// launches GPU work on the stream. CUDA graph capture can't include host
// memcpy, so for graph-based detectors we need to split the two halves:
//
//   every frame:
//     gpu_preprocess_stage_host(src_u8, src_w*src_h*3);  // plain host memcpy
//     cudaGraphLaunch(exec, stream);                     // replay captured GPU ops
//
// where the captured graph contains exactly one invocation of
// gpu_preprocess_run_gpu + model inference + output DMA.
//
// `LetterboxInfo` is deterministic from (src_w, src_h, dst_w, dst_h), so
// callers can compute it once at graph-build time.

void gpu_preprocess_stage_host(const uint8_t* src_u8, int src_bytes);

void gpu_preprocess_run_gpu(int src_w, int src_h,
                            float* dst_f32_device, int dst_w, int dst_h,
                            const PreprocessParams& params,
                            cudaStream_t stream);

LetterboxInfo gpu_preprocess_compute_letterbox(int src_w, int src_h,
                                               int dst_w, int dst_h);

}  // namespace booster_vision
