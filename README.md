# vision

YOLOv8-based object detection for RoboCup Humanoid Soccer using NVIDIA TensorRT.

## License

Apache-2.0

## Attribution

This package is based on the vision module from [Booster Robotics' robocupdemo](https://github.com/BoosterRobotics/robocupdemo) (branch: `sandbox/feat/k1_3v3_demo`).

- Original: Copyright 2024 Booster Robotics
- Minor modifications: Copyright 2025 RFC-Tsudanuma

### Modifications from original

- `launch/launch.py`: Disabled segmentation and depth by default for performance optimization
- `launch/launch_optimized.py`: New file for CPU affinity configuration
- `src/vision_node.cpp`: Added null check for data_logger

