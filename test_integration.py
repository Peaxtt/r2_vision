#!/usr/bin/env python3
"""Headless ROS2 + OpenVINO integration test for the weapon pipeline.

Runs WITHOUT a RealSense camera:
  1. load the real weapon OpenVINO IR via model_loader
  2. run YOLO inference on a test frame
  3. push results through WeaponProcessor on a REAL rclpy node
  4. a subscriber node receives /vision/task1_target over real DDS

Usage:
  python3 test_integration.py [optional_image.png]
"""

import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from model_loader import load_model
from processors import WeaponProcessor
from yolo_node import MODEL_WEAPON, CENTROID_CFG

RT_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST, depth=1)


class HostNode(Node):
    """Minimal real node providing what WeaponProcessor needs from `node`."""

    def __init__(self):
        super().__init__('r2_vision_test')
        self.declare_parameter('centroid_config', CENTROID_CFG)
        self.declare_parameter('max_depth_mm', 3400)
        self._pubs = {}

    def get_publisher(self, topic, msg_type, reliable=False):
        if topic not in self._pubs:
            self._pubs[topic] = self.create_publisher(msg_type, topic, RT_QOS)
        return self._pubs[topic]


class EchoNode(Node):
    def __init__(self):
        super().__init__('r2_vision_test_echo')
        self.got = None
        self.create_subscription(String, '/vision/task1_target',
                                 self._cb, RT_QOS)

    def _cb(self, msg):
        self.got = msg.data
        self.get_logger().info(f'RECEIVED over DDS:\n{msg.data}')


def main():
    rclpy.init()

    # --- 1. load real OpenVINO model ---
    print('[1] Loading weapon OpenVINO IR ...')
    model = load_model(MODEL_WEAPON, log=print)
    assert model is not None, f'model not found: {MODEL_WEAPON}'
    print(f'    classes = {model.names}')

    # --- 2. inference on a frame ---
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
        assert frame is not None, f'cannot read {sys.argv[1]}'
        print(f'[2] Inference on {sys.argv[1]} {frame.shape}')
    else:
        frame = np.full((480, 640, 3), 127, np.uint8)
        print('[2] Inference on blank 640x480 frame (no weapon → 0 detections expected)')
    depth = np.full(frame.shape[:2], 1000, np.uint16)

    results = model.predict(frame, imgsz=640, conf=0.5, device='CPU', verbose=False)
    n = sum(0 if r.boxes is None else len(r.boxes) for r in results)
    print(f'    raw YOLO boxes = {n}')

    # --- 3. real node + processor ---
    host = HostNode()
    echo = EchoNode()
    spin = threading.Thread(
        target=lambda: rclpy.spin(echo), daemon=True)
    spin.start()
    time.sleep(0.5)  # discovery

    fx = fy = 600.0
    intr = (fx, fy, frame.shape[1] / 2, frame.shape[0] / 2)
    wp = WeaponProcessor(host, 'WEAPON_CLUB', '/vision/task1_target')

    print('[3] WeaponProcessor.process() → publish /vision/task1_target')
    wp.process(results, frame, depth, frame.copy(), intr)

    # --- 4. confirm delivery ---
    for _ in range(50):
        if echo.got is not None:
            break
        time.sleep(0.05)

    print('\n[4] RESULT:')
    if echo.got is not None:
        print('    ✓ message delivered over real ROS2 DDS')
    else:
        print('    ✗ no message received')
        rclpy.shutdown()
        sys.exit(1)

    host.destroy_node()
    echo.destroy_node()
    rclpy.shutdown()
    print('\nINTEGRATION TEST PASSED')


if __name__ == '__main__':
    main()
