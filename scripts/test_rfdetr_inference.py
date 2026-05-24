#!/usr/bin/env python3
"""Test RF-DETR inference with Red/Blue → K1 color assignment.

Replicates the C++ vision_node logic in Python for visual verification.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from rfdetr import RFDETRNano

CLASSNAMES = ["Ball", "Goalpost", "K1", "LCross", "PenaltyPoint",
              "TCross", "XCross", "Red", "Blue"]

CONF_THRESHOLDS = {
    "Ball": 0.4, "Goalpost": 0.4, "K1": 0.4,
    "LCross": 0.3, "PenaltyPoint": 0.3, "TCross": 0.3, "XCross": 0.3,
    "Red": 0.3, "Blue": 0.3,
}

COLORS = {
    "Ball":         (0, 255, 255),
    "Goalpost":     (255, 255, 0),
    "K1":           (0, 255, 0),
    "LCross":       (200, 200, 200),
    "PenaltyPoint": (200, 200, 200),
    "TCross":       (200, 200, 200),
    "XCross":       (200, 200, 200),
    "Red":          (0, 0, 255),
    "Blue":         (255, 0, 0),
}


def main():
    ckpt = Path("src/vision/model/checkpoint_best_ema.pth")
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src/vision/2302.jpg")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else img_path.with_name(img_path.stem + "_result.jpg")

    model = RFDETRNano(pretrain_weights=str(ckpt), num_classes=len(CLASSNAMES))
    detections = model.predict(str(img_path), threshold=0.2)

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    dets = []
    for xyxy, conf, cls_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        cls_id = int(cls_id)
        if cls_id < 0 or cls_id >= len(CLASSNAMES):
            continue
        name = CLASSNAMES[cls_id]
        c = float(conf)
        if c < CONF_THRESHOLDS.get(name, 0.2):
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        dets.append({"name": name, "conf": c, "box": (x1, y1, x2, y2), "color": ""})

    # Separate Red/Blue from others
    color_dets = [d for d in dets if d["name"] in ("Red", "Blue")]
    remaining = [d for d in dets if d["name"] not in ("Red", "Blue")]

    # Assign Red/Blue to K1 by center-inside check
    for det in remaining:
        if det["name"] != "K1":
            continue
        kx1, ky1, kx2, ky2 = det["box"]
        best_conf = -1.0
        best_color = ""
        for cd in color_dets:
            cx = (cd["box"][0] + cd["box"][2]) // 2
            cy = (cd["box"][1] + cd["box"][3]) // 2
            if kx1 <= cx <= kx2 and ky1 <= cy <= ky2:
                if cd["conf"] > best_conf:
                    best_conf = cd["conf"]
                    best_color = "red" if cd["name"] == "Red" else "blue"
        if best_conf >= 0:
            det["color"] = best_color

    # --- Draw ---
    # First pass: draw Red/Blue raw bboxes (thin, dashed-style)
    for d in color_dets:
        x1, y1, x2, y2 = d["box"]
        c = COLORS[d["name"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 1)
        label = f'{d["name"]} {d["conf"]:.2f}'
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

    # Second pass: draw remaining detections
    for d in remaining:
        x1, y1, x2, y2 = d["box"]
        name = d["name"]
        c = COLORS.get(name, (255, 255, 255))

        if name == "K1" and d["color"]:
            # Override K1 box color based on assigned team color
            if d["color"] == "red":
                c = (0, 0, 255)
            else:
                c = (255, 0, 0)
            label = f'K1 [{d["color"]}] {d["conf"]:.2f}'
        else:
            label = f'{name} {d["conf"]:.2f}'

        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), c, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)

    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")

    # Print summary (simulates what would be published as DetectedObject messages)
    print("\n=== Published Detections (Red/Blue excluded) ===")
    for d in remaining:
        color_str = f'  color="{d["color"]}"' if d["color"] else ""
        print(f'  label="{d["name"]}"  conf={d["conf"]:.3f}  '
              f'bbox={d["box"]}{color_str}')

    print(f"\n=== Raw Red/Blue Detections (not published, used for matching) ===")
    for d in color_dets:
        print(f'  label="{d["name"]}"  conf={d["conf"]:.3f}  bbox={d["box"]}')


if __name__ == "__main__":
    raise SystemExit(main())
