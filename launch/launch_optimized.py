#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024 Booster Robotics
# SPDX-FileCopyrightText: 2025 RFC-Tsudanuma (minor modifications)
# SPDX-License-Identifier: Apache-2.0
#
# Based on robocupdemo vision package by Booster Robotics
#
# [Contributors]
# - Ikuo Shige

"""
Optimized launch file for vision with CPU affinity.

This launch file configures vision for balanced performance on Jetson AGX Orin
by setting:
1. CPU affinity to cores 6-11 (separate from whiteline_scan)
2. Segmentation disabled for performance
3. Data saving disabled
4. Depth processing disabled

Usage:
    # Standard mode
    ros2 launch vision launch_optimized.py

    # With custom config
    ros2 launch vision launch_optimized.py vision_config_path:=/path/to/config
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def handle_configuration(context, *args, **kwargs):
    default_config_path = PathJoinSubstitution([FindPackageShare('vision'), 'config']).perform(context)

    user_cfg_dir = LaunchConfiguration('vision_config_path').perform(context)
    config_path = default_config_path
    if user_cfg_dir and user_cfg_dir.strip():
        cand = user_cfg_dir.rstrip('/')
        if os.path.exists(os.path.join(cand, 'vision.yaml')):
            config_path = cand
        else:
            print(f"[vision launch] warning: {cand}/vision.yaml not found, fallback to {default_config_path}")
    config_file = os.path.join(config_path, 'vision.yaml')
    config_local_file = os.path.join(config_path, 'vision_local.yaml')

    show_det = LaunchConfiguration('show_det')
    show_seg = LaunchConfiguration('show_seg')
    save_data = LaunchConfiguration('save_data')
    save_depth = LaunchConfiguration('save_depth')
    offline_mode = LaunchConfiguration('offline_mode')
    save_fps = LaunchConfiguration('save_fps')
    detection_model_path = LaunchConfiguration('detection_model_path')
    segmentation_model_path = LaunchConfiguration('segmentation_model_path')
    camera_type = LaunchConfiguration('camera_type')
    cpu_affinity = LaunchConfiguration('cpu_affinity')

    # Build prefix command for CPU affinity
    prefix_cmd = PythonExpression([
        "'taskset -c ", cpu_affinity, "'"
    ])

    return [
        Node(
            package='vision',
            executable='vision_node',
            name='vision_node',
            output='screen',
            arguments=[config_file, config_local_file],
            parameters=[{
                'offline_mode': offline_mode,
                'show_det': show_det,
                'show_seg': show_seg,
                'save_data': save_data,
                'save_depth': save_depth,
                'save_fps': save_fps,
                'detection_model_path': detection_model_path,
                'segmentation_model_path': segmentation_model_path,
                'camera_type': camera_type
            }],
            prefix=prefix_cmd,
            emulate_tty=True
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'vision_config_path',
            default_value='',
            description='Optional directory containing vision.yaml & vision_local.yaml (empty => use package default)'
        ),
        DeclareLaunchArgument(
            "offline_mode",
            default_value='false',
            description="enable offline model"
        ),
        DeclareLaunchArgument(
            "show_det",
            default_value='false',
            description="Show detection result"
        ),
        DeclareLaunchArgument(
            "show_seg",
            default_value='false',
            description="Show segmentation result"
        ),
        DeclareLaunchArgument(
            "save_data",
            default_value='false',  # disabled for performance optimization
            description="Save recevied image data"
        ),
        DeclareLaunchArgument(
            "save_depth",
            default_value='false',  # disabled for performance optimization
            description="Save recevied depth img data"
        ),
        DeclareLaunchArgument(
            "save_fps",
            default_value='0',  # disabled for performance optimization
            description="Save n frames of data each second"
        ),
        DeclareLaunchArgument(
            'detection_model_path',
            default_value="",
            description="param to override detection_model.model_path. will not override if empty"
        ),
        DeclareLaunchArgument(
            'segmentation_model_path',
            default_value="",  # disabled for performance optimization
            description="param to override segmentation_model.model_path. will not override if empty"
        ),
        DeclareLaunchArgument(
            'camera_type',
            default_value="",
            description="param to override camera.type. will not override if empty"
        ),
        DeclareLaunchArgument(
            'cpu_affinity',
            default_value='6-11',
            description='CPU cores to bind to (separate from whiteline_scan which uses 0-5)'
        ),
        OpaqueFunction(function=handle_configuration)
    ])
