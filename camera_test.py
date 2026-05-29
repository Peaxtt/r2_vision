#!/usr/bin/env python3
"""
Standalone camera test — RealSense + OpenCV window
เทสกล้องโดดๆ ไม่มี ROS ไม่มี YOLO
กด Q หรือ ESC เพื่อออก
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import time

WIDTH, HEIGHT, FPS = 848, 480, 30

p = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FPS)

print(f"Starting RealSense {WIDTH}x{HEIGHT}@{FPS}fps ...")
p.start(cfg)
align = rs.align(rs.stream.color)
print("Warming up (may take ~15s)...")

cv2.namedWindow("Camera Test — press Q to quit", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Test — press Q to quit", WIDTH, HEIGHT)

frame_count = 0
t0 = time.time()

try:
    while True:
        try:
            frames  = p.wait_for_frames(timeout_ms=5000)
        except Exception:
            print("  timeout — retrying...")
            continue

        aligned     = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame:
            continue

        img   = np.asanyarray(color_frame.get_data())
        frame_count += 1
        fps   = frame_count / (time.time() - t0)

        # Depth at center
        cx, cy = WIDTH // 2, HEIGHT // 2
        d = depth_frame.get_distance(cx, cy) if depth_frame else 0.0

        cv2.putText(img, f"FPS: {fps:.1f}  Depth: {d:.3f}m",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        cv2.imshow("Camera Test — press Q to quit", img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
finally:
    p.stop()
    cv2.destroyAllWindows()
    print(f"Done — {frame_count} frames @ avg {frame_count/(time.time()-t0):.1f}fps")
