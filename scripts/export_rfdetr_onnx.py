#!/usr/bin/env python3
"""Export an RF-DETR Nano checkpoint to ONNX.

RF-DETR does not ship an ultralytics-style `.export(format="engine")` path, so
we produce an ONNX file first; `build_rfdetr_trt.sh` turns that into a
TensorRT engine on the target GPU.

The exported ONNX exposes RAW outputs (pred_logits + pred_boxes); sigmoid +
top-k + cxcywh->xyxy is done in the C++ RFDetrDetectorTRT runtime.

Example:
    python3 export_rfdetr_onnx.py \\
        --checkpoint /path/to/checkpoint_best_total.pth \\
        --output     src/vision/model/rfdetr_nano_384.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export(checkpoint: Path, output: Path, imgsz: int, opset: int,
           simplify: bool) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from rfdetr import RFDETRNano  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "rfdetr package not installed. Install with: "
            "pip install 'rfdetr[train]>=1.6.4'"
        ) from e

    print(f"[export_rfdetr_onnx] checkpoint: {checkpoint}")
    print(f"[export_rfdetr_onnx] output    : {output}")
    print(f"[export_rfdetr_onnx] imgsz     : {imgsz}")
    print(f"[export_rfdetr_onnx] opset     : {opset}")

    model = RFDETRNano(pretrain_weights=str(checkpoint))
    # rfdetr exposes an `.export()` method that writes ONNX; the API varies by
    # version, so we try a few common signatures before falling back to torch.
    export_fn = getattr(model, "export", None)
    if export_fn is not None:
        try:
            export_fn(output_dir=str(output.parent), opset_version=opset,
                      simplify=simplify)
        except TypeError:
            # older rfdetr versions: export(dir)
            export_fn(str(output.parent))
        produced = sorted(output.parent.glob("*.onnx"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not produced:
            raise RuntimeError("rfdetr .export produced no .onnx file")
        if produced[0].resolve() != output.resolve():
            produced[0].rename(output)
    else:
        raise RuntimeError(
            "Installed rfdetr has no .export API; please upgrade to >=1.6.4 "
            "or implement a manual torch.onnx.export wrapper.")

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnxsim_simplify  # type: ignore
            model_sim, ok = onnxsim_simplify(onnx.load(str(output)))
            if ok:
                onnx.save(model_sim, str(output))
                print("[export_rfdetr_onnx] onnx-simplifier ok")
        except ImportError:
            print("[export_rfdetr_onnx] onnxsim not installed, skipping simplify")

    return output


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("src/vision/model/rfdetr_nano_384.onnx"))
    p.add_argument("--imgsz", type=int, default=384)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-simplify", dest="simplify", action="store_false")
    p.set_defaults(simplify=True)
    args = p.parse_args()

    if not args.checkpoint.exists():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    export(args.checkpoint, args.output, args.imgsz, args.opset, args.simplify)
    print("[export_rfdetr_onnx] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
