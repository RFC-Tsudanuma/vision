// SPDX-FileCopyrightText: 2026 RFC-Tsudanuma
// SPDX-License-Identifier: Apache-2.0

#include "booster_vision/model/trt/rfdetr_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <opencv2/opencv.hpp>

#include "booster_vision/model/trt/gpu_preprocess.h"
#include "booster_vision/model/trt/logging.h"

namespace booster_vision {

namespace {

Logger& GetLogger() {
    static Logger l;
    return l;
}

size_t DimsVolume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] > 0) v *= static_cast<size_t>(d.d[i]);
    }
    return v;
}

inline float Sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

}  // namespace

bool RFDetrDetectorTRT::LoadEngine() {
    std::ifstream f(model_path_, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const size_t n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(n);
    f.read(buf.data(), n);

    runtime_.reset(nvinfer1::createInferRuntime(GetLogger()));
    if (!runtime_) return false;
    engine_.reset(runtime_->deserializeCudaEngine(buf.data(), buf.size()),
                  [](nvinfer1::ICudaEngine* p) { delete p; });
    if (!engine_) return false;

    const int n_io = engine_->getNbIOTensors();
    if (n_io != 3) {
        std::cerr << "RFDetrDetectorTRT: expected 3 IO tensors (input, logits, "
                     "boxes), got "
                  << n_io << std::endl;
        return false;
    }

    // Identify tensors by shape: input is 4D (NCHW), logits is 3D (NxQxC),
    // boxes is 3D (NxQx4). This avoids relying on naming conventions which
    // differ across rfdetr versions.
    for (int i = 0; i < n_io; ++i) {
        const char* name = engine_->getIOTensorName(i);
        const nvinfer1::Dims d = engine_->getTensorShape(name);
        const nvinfer1::TensorIOMode io = engine_->getTensorIOMode(name);
        if (io == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
            if (d.nbDims >= 4) {
                input_h_ = d.d[2];
                input_w_ = d.d[3];
            }
            input_size_floats_ = DimsVolume(d);
        } else {
            if (d.nbDims == 3 && d.d[2] == 4) {
                boxes_name_ = name;
                num_queries_ = d.d[1];
                boxes_size_floats_ = DimsVolume(d);
            } else if (d.nbDims == 3) {
                logits_name_ = name;
                num_queries_ = d.d[1];
                num_classes_ = d.d[2];
                logits_size_floats_ = DimsVolume(d);
            }
        }
    }

    if (input_name_.empty() || logits_name_.empty() || boxes_name_.empty()) {
        std::cerr << "RFDetrDetectorTRT: failed to identify IO tensors"
                  << std::endl;
        return false;
    }

    std::cout << "RF-DETR engine loaded: input=" << input_w_ << "x" << input_h_
              << " queries=" << num_queries_ << " classes=" << num_classes_
              << std::endl;
    return true;
}

void RFDetrDetectorTRT::Init(std::string model_path) {
    if (model_path.find(".engine") == std::string::npos) {
        throw std::runtime_error("RFDetrDetectorTRT: not an .engine file: " +
                                 model_path);
    }
    if (!LoadEngine()) {
        throw std::runtime_error("RFDetrDetectorTRT: failed to load engine: " +
                                 model_path);
    }

    // Pinned host memory for fast DMA on Jetson (cudaMallocHost is critical
    // when Tegra shares GPU/CPU memory via SMMU; page-locked is 3-4x faster
    // than pageable for outbound D2H copies).
    cudaMallocHost((void**)&host_input_, input_size_floats_ * sizeof(float));
    cudaMallocHost((void**)&host_logits_, logits_size_floats_ * sizeof(float));
    cudaMallocHost((void**)&host_boxes_, boxes_size_floats_ * sizeof(float));
    cudaMalloc(&dev_input_, input_size_floats_ * sizeof(float));
    cudaMalloc(&dev_logits_, logits_size_floats_ * sizeof(float));
    cudaMalloc(&dev_boxes_, boxes_size_floats_ * sizeof(float));
    cudaStreamCreate(&stream_);
    gpu_preprocess_init(4096 * 4096);  // enough for stereo 1344x376 or larger.

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error(
            "RFDetrDetectorTRT: failed to create execution context");
    }
    context_->setTensorAddress(input_name_.c_str(), dev_input_);
    context_->setTensorAddress(logits_name_.c_str(), dev_logits_);
    context_->setTensorAddress(boxes_name_.c_str(), dev_boxes_);
}

RFDetrDetectorTRT::~RFDetrDetectorTRT() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (stream_) cudaStreamDestroy(stream_);
    if (dev_input_) cudaFree(dev_input_);
    if (dev_logits_) cudaFree(dev_logits_);
    if (dev_boxes_) cudaFree(dev_boxes_);
    if (host_input_) cudaFreeHost(host_input_);
    if (host_logits_) cudaFreeHost(host_logits_);
    if (host_boxes_) cudaFreeHost(host_boxes_);
}

// Capture the GPU side of inference into a CUDA graph: pinned-to-device DMA,
// preprocess kernel, enqueueV3, output DMAs. Returns true on success. The
// host-side memcpy into g_host_staging stays OUTSIDE the graph.
bool RFDetrDetectorTRT::CaptureGraph(int src_w, int src_h) {
    PreprocessParams params;
    params.mean[0] = 0.485f; params.mean[1] = 0.456f; params.mean[2] = 0.406f;
    params.std[0]  = 0.229f; params.std[1]  = 0.224f; params.std[2]  = 0.225f;
    params.pad_value = 0;
    params.bgr_to_rgb = true;

    // Priming run so that any lazy-initialized TRT buffers are touched before
    // capture; otherwise capture may record spurious cuBLAS/cuDNN setup work.
    gpu_preprocess_run_gpu(src_w, src_h, static_cast<float*>(dev_input_),
                           input_w_, input_h_, params, stream_);
    context_->enqueueV3(stream_);
    cudaMemcpyAsync(host_logits_, dev_logits_,
                    logits_size_floats_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(host_boxes_, dev_boxes_,
                    boxes_size_floats_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    cudaGraph_t graph;
    if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
        return false;
    }

    gpu_preprocess_run_gpu(src_w, src_h, static_cast<float*>(dev_input_),
                           input_w_, input_h_, params, stream_);
    context_->enqueueV3(stream_);
    cudaMemcpyAsync(host_logits_, dev_logits_,
                    logits_size_floats_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(host_boxes_, dev_boxes_,
                    boxes_size_floats_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);

    const auto capture_err = cudaStreamEndCapture(stream_, &graph);
    if (capture_err != cudaSuccess) {
        std::cerr << "RFDetr: cudaStreamEndCapture failed: "
                  << cudaGetErrorString(capture_err) << std::endl;
        return false;
    }

    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    const auto inst_err = cudaGraphInstantiate(&graph_exec_, graph, 0ull);
    cudaGraphDestroy(graph);
    if (inst_err != cudaSuccess) {
        graph_exec_ = nullptr;
        std::cerr << "RFDetr: cudaGraphInstantiate failed: "
                  << cudaGetErrorString(inst_err) << std::endl;
        return false;
    }
    graph_src_w_ = src_w;
    graph_src_h_ = src_h;
    return true;
}

std::vector<DetectionRes> RFDetrDetectorTRT::Inference(const cv::Mat& img) {
    const int sw = img.cols;
    const int sh = img.rows;
    const int src_bytes = sw * sh * 3;
    const LetterboxInfo lb =
        gpu_preprocess_compute_letterbox(sw, sh, input_w_, input_h_);

    // (Re)capture when source size changes — only if graph replay is enabled.
    if (use_cuda_graph_ &&
        (!graph_exec_ || sw != graph_src_w_ || sh != graph_src_h_)) {
        if (graph_exec_) {
            cudaGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
        std::cerr << "[RFDetr] capturing CUDA graph for " << sw << "x" << sh << std::endl;
        CaptureGraph(sw, sh);  // best-effort; falls back to non-graph on failure
    }

    if (use_cuda_graph_ && graph_exec_) {
        gpu_preprocess_stage_host(img.ptr<uint8_t>(), src_bytes);
        cudaGraphLaunch(graph_exec_, stream_);
        cudaStreamSynchronize(stream_);
    } else {
        PreprocessParams params;
        params.mean[0] = 0.485f; params.mean[1] = 0.456f; params.mean[2] = 0.406f;
        params.std[0]  = 0.229f; params.std[1]  = 0.224f; params.std[2]  = 0.225f;
        params.pad_value = 0;
        params.bgr_to_rgb = true;
        gpu_preprocess_letterbox(img.ptr<uint8_t>(), sw, sh,
                                 static_cast<float*>(dev_input_), input_w_, input_h_,
                                 params, stream_, nullptr);
        if (!context_->enqueueV3(stream_)) return {};
        cudaMemcpyAsync(host_logits_, dev_logits_,
                        logits_size_floats_ * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(host_boxes_, dev_boxes_,
                        boxes_size_floats_ * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    }

    return PostProcess(sw, sh, lb.scale, lb.dx, lb.dy);
}

std::vector<DetectionRes> RFDetrDetectorTRT::PostProcess(
    int src_w, int src_h, float scale, int dx, int dy) {
    std::vector<DetectionRes> out;
    if (num_queries_ <= 0 || num_classes_ <= 0) return out;
    out.reserve(16);

    // Sigmoid is monotonic, so compare against the logit threshold directly
    // and skip sigmoid for rejected queries. Only compute sigmoid on the
    // winner's logit to fill in `confidence`. For conf=0.25 the logit
    // threshold is log(0.25/0.75) ~= -1.0986.
    const float logit_thresh = std::log(confidence_ / (1.0f - confidence_));
    const float inv_scale = 1.0f / scale;

    for (int q = 0; q < num_queries_; ++q) {
        const float* logits = host_logits_ + q * num_classes_;
        int best_cls = 0;
        float best_logit = logits[0];
        for (int c = 1; c < num_classes_; ++c) {
            if (logits[c] > best_logit) { best_logit = logits[c]; best_cls = c; }
        }
        if (best_logit < logit_thresh) continue;

        const float* b = host_boxes_ + q * 4;  // cxcywh normalized [0,1]
        const float cx = b[0] * input_w_;
        const float cy = b[1] * input_h_;
        const float half_bw = 0.5f * b[2] * input_w_;
        const float half_bh = 0.5f * b[3] * input_h_;

        const float x1 = ((cx - half_bw) - dx) * inv_scale;
        const float y1 = ((cy - half_bh) - dy) * inv_scale;
        const float x2 = ((cx + half_bw) - dx) * inv_scale;
        const float y2 = ((cy + half_bh) - dy) * inv_scale;

        const int xmin = std::clamp(static_cast<int>(x1), 0, src_w - 1);
        const int ymin = std::clamp(static_cast<int>(y1), 0, src_h - 1);
        const int xmax = std::clamp(static_cast<int>(x2), 0, src_w - 1);
        const int ymax = std::clamp(static_cast<int>(y2), 0, src_h - 1);
        if (xmax - xmin < 2 || ymax - ymin < 2) continue;

        DetectionRes d;
        d.bbox = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
        d.confidence = Sigmoid(best_logit);
        d.class_id = best_cls;
        out.push_back(d);
    }
    // No NMS: RF-DETR uses set-prediction (Hungarian matching) so duplicates
    // are extremely rare in practice. The previous NMS pass cost ~40us in
    // postprocess and rarely removed anything.
    return out;
}

}  // namespace booster_vision
