// SPDX-FileCopyrightText: 2026 RFC-Tsudanuma
// SPDX-License-Identifier: Apache-2.0
//
// RF-DETR Nano TensorRT detector. Subclasses YoloV8Detector for polymorphism
// (same interface, same DetectionRes output); naming is historical — think of
// it as "a TRT-backed object detector" rather than a YOLOv8 variant.

#pragma once

#include <string>
#include <vector>

#include <NvInfer.h>

#include "booster_vision/model/detector.h"

namespace booster_vision {

class RFDetrDetectorTRT : public YoloV8Detector {
 public:
    explicit RFDetrDetectorTRT(const std::string& engine_path)
        : YoloV8Detector(engine_path) {
        Init(engine_path);
    }
    ~RFDetrDetectorTRT() override;

    void Init(std::string model_path) override;
    std::vector<DetectionRes> Inference(const cv::Mat& img) override;

    // Enable cudaGraph-based replay of the GPU pipeline. Reduces launch
    // overhead by ~50us on desktop GPUs and ~200us on Jetson Orin Nano. On
    // some consumer GPUs with aggressive clock gating this can hurt tail
    // latency (p99), so it is OFF by default. Turn it on for Jetson.
    void EnableCudaGraph(bool on) { use_cuda_graph_ = on; }

 private:
    bool LoadEngine();
    std::vector<DetectionRes> PostProcess(int src_w, int src_h,
                                          float letterbox_scale,
                                          int letterbox_dx,
                                          int letterbox_dy);

    std::shared_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = 0;

    int input_h_ = 384;
    int input_w_ = 384;
    int num_queries_ = 0;
    int num_classes_ = 0;
    std::string input_name_;
    std::string logits_name_;
    std::string boxes_name_;

    size_t input_size_floats_ = 0;
    size_t logits_size_floats_ = 0;
    size_t boxes_size_floats_ = 0;

    float* host_input_ = nullptr;
    float* host_logits_ = nullptr;
    float* host_boxes_ = nullptr;
    void* dev_input_ = nullptr;
    void* dev_logits_ = nullptr;
    void* dev_boxes_ = nullptr;

    // CUDA Graph replay (opt-in via EnableCudaGraph(true)). Saves CUDA launch
    // overhead by replaying a captured sequence instead of re-issuing each
    // API call. Large win on Jetson (weak CPU), marginal-to-negative on
    // desktop RTX where power/clock gating introduces ~3ms wakeup tails.
    bool use_cuda_graph_ = false;
    cudaGraphExec_t graph_exec_ = nullptr;
    int graph_src_w_ = 0;  // src width that was captured; invalidate on change
    int graph_src_h_ = 0;
    bool CaptureGraph(int src_w, int src_h);
};

}  // namespace booster_vision
