#!/usr/bin/env bash
# Build an RF-DETR TensorRT engine from an exported ONNX file.
# Must be run on the target GPU (x86 RTX or Jetson Orin Nano) because TRT
# engines are not portable across GPU generations.

set -euo pipefail

ONNX_PATH=${1:-src/vision/model/rfdetr_nano_384.onnx}
ENGINE_PATH=${2:-src/vision/model/rfdetr_nano_384_fp16.engine}
WORKSPACE_MB=${3:-4096}

if ! command -v trtexec >/dev/null 2>&1; then
  echo "error: trtexec not found in PATH." >&2
  echo "       On x86 install NVIDIA TensorRT; on Jetson it ships with JetPack." >&2
  exit 1
fi

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "error: ONNX file missing: $ONNX_PATH" >&2
  echo "       Run export_rfdetr_onnx.py first." >&2
  exit 2
fi

mkdir -p "$(dirname "$ENGINE_PATH")"

echo "[build_rfdetr_trt] onnx   : $ONNX_PATH"
echo "[build_rfdetr_trt] engine : $ENGINE_PATH"
echo "[build_rfdetr_trt] workspace: ${WORKSPACE_MB} MB"

trtexec \
  --onnx="$ONNX_PATH" \
  --saveEngine="$ENGINE_PATH" \
  --fp16 \
  --memPoolSize=workspace:${WORKSPACE_MB} \
  --verbose \
  --useCudaGraph

echo "[build_rfdetr_trt] done -> $ENGINE_PATH"
