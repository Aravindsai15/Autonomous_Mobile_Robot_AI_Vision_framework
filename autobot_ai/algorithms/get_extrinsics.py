#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np

# Create a pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable both depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get stream profiles
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()

# Get intrinsics
depth_intrinsics = depth_stream.get_intrinsics()
color_intrinsics = color_stream.get_intrinsics()

# Get extrinsics from depth to color
depth_to_color_extrinsics = depth_stream.get_extrinsics_to(color_stream)

# Stop the pipeline
pipeline.stop()

# Print extrinsics
print("\n=== Extrinsics: Depth to Color ===")
R = np.array(depth_to_color_extrinsics.rotation).reshape(3, 3)
T = np.array(depth_to_color_extrinsics.translation)
print("Rotation matrix (3x3):\n", R)
print("Translation vector (in meters):\n", T)

# Print intrinsics manually
def print_intrinsics(name, intr):
    print(f"\n=== {name} Intrinsics ===")
    print(f"Resolution: {intr.width}x{intr.height}")
    print(f"Principal Point: ({intr.ppx:.2f}, {intr.ppy:.2f})")
    print(f"Focal Length: (fx={intr.fx:.2f}, fy={intr.fy:.2f})")
    print(f"Distortion Model: {intr.model}")
    print(f"Distortion Coeffs: {intr.coeffs}")

print_intrinsics("Depth", depth_intrinsics)
print_intrinsics("Color", color_intrinsics)
