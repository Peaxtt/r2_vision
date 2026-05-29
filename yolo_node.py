#!/usr/bin/env python3
"""
YOLO Detection ROS2 Node — ABU Robocon 2026 R2
กล้อง + OpenVINO YOLO + OpenCV window + ROS2 publisher

Each macro_state runs its OWN processing pipeline (see processors.py):
  WEAPON_CLUB              → WeaponProcessor   (full OpenVINO weapon pipeline,
                             mm coords + placement index, JSON output)
  MEIHUA_FOREST_EXECUTION  → SimpleBestProcessor (pixel-error PoseStamped)
  MARTIAL_ART_PLACEMENT    → AbstractProcessor (no model yet — placeholder)

Publishes:
  /vision/task1_target  String JSON  BEST_EFFORT  weapon detections (mm + index)
  /vision/task2_target  PoseStamped  BEST_EFFORT  pixel error + depth (mm)
  /vision/task3_target  —            (abstract, no model)

Subscribes:
  /system_status    String JSON  BEST_EFFORT  macro_state → switch model
  /weapon_selection String JSON  RELIABLE     runtime class switch (fallback)

Display: OpenCV window บน main thread (ไม่ค้าง)
Models:  ONE OpenVINO model loaded at a time (unload เก่าก่อน load ใหม่)
"""

import gc
import json
import os
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

import cv2
import numpy as np
import pyrealsense2 as rs

from model_loader import load_model
from processors import WeaponProcessor, SimpleBestProcessor, GridProcessor

# ── QoS ───────────────────────────────────────────────────────────────────────
RT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST, depth=1)
RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST, depth=10)

# ── State → model parameter key ───────────────────────────────────────────────
STATE_MODEL = {
    'WEAPON_CLUB':             'model_weapon',
    'MEIHUA_FOREST_EXECUTION': 'model_meihua',
    'MARTIAL_ART_PLACEMENT':   'model_martial',
}

# ── Model paths (เปลี่ยนตรงนี้เมื่อ pull model ใหม่) ─────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, 'models')
# Weapon: pre-converted OpenVINO IR, committed under models/.
MODEL_WEAPON  = os.path.join(MODELS_DIR, 'yolonew_openvino_model')
# Cube (blue_cube/red_cube) — shared by the forest (task2) and grid (task3)
# tasks. Committed as OpenVINO IR. .pt/.onnx sources auto-export on first load.
MODEL_MEIHUA  = os.path.join(MODELS_DIR, 'cube_openvino_model')
MODEL_MARTIAL = os.path.join(MODELS_DIR, 'cube_openvino_model')
CENTROID_CFG  = os.path.join(MODELS_DIR, 'centroid_config.txt')

YOLO_FPS   = 5    # YOLO inference rate cap
CAM_W      = 848
CAM_H      = 480
CAM_FPS    = 30
WIN_NAME   = 'R2 Vision — Q to quit'


# ── Rotation helpers ──────────────────────────────────────────────────────────

def rotate_frame(frame, deg):
    deg = int(deg) % 360
    if deg == 0:   return frame
    if deg == 90:  return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def rotate_intrinsics(fx, fy, ppx, ppy, W, H, deg):
    deg = int(deg) % 360
    if deg == 90:  return fy, fx, H - 1 - ppy, ppx
    if deg == 180: return fx, fy, W - 1 - ppx, H - 1 - ppy
    if deg == 270: return fy, fx, ppy, W - 1 - ppx
    return fx, fy, ppx, ppy


class VisionNode(Node):

    def __init__(self):
        super().__init__('r2_vision_node')
        self._cb = ReentrantCallbackGroup()

        # Parameters (override via --ros-args -p)
        self.declare_parameter('model_weapon',   MODEL_WEAPON)
        self.declare_parameter('model_meihua',   MODEL_MEIHUA)
        self.declare_parameter('model_martial',  MODEL_MARTIAL)
        self.declare_parameter('centroid_config', CENTROID_CFG)
        self.declare_parameter('conf',           0.50)
        self.declare_parameter('imgsz',          640)
        self.declare_parameter('device',         'CPU')   # OpenVINO: CPU/GPU/AUTO
        self.declare_parameter('rotate',         0)        # 0/90/180/270 clockwise
        self.declare_parameter('target_weapon',  'spearhead')
        self.declare_parameter('target_meihua',  '')      # '' = ทุก class
        self.declare_parameter('grid_use_depth',  True)    # grid: depth-confirm cells
        self.declare_parameter('cam_width',      CAM_W)
        self.declare_parameter('cam_height',     CAM_H)
        self.declare_parameter('cam_fps',        CAM_FPS)
        self.declare_parameter('max_depth_mm',   3400)

        # State
        self._state   = 'IDLE'
        self._tgt_cls = self.get_parameter('target_weapon').value
        self._lock    = threading.Lock()
        self._running = True
        self._rotate  = int(self.get_parameter('rotate').value)

        # Camera queue
        self._q: queue.Queue = queue.Queue(maxsize=1)

        # Display buffer (main thread reads this)
        self._frame     = None
        self._frame_lk  = threading.Lock()

        # ONE active model
        self._model     = None
        self._model_key = ''
        self._model_lk  = threading.Lock()

        # Publishers cache (topic → publisher)
        self._pubs: dict = {}

        # Camera params
        self._w   = self.get_parameter('cam_width').value
        self._h   = self.get_parameter('cam_height').value
        self._fps = self.get_parameter('cam_fps').value
        self._fx = self._fy = 0.0
        self._ppx = self._w / 2.0
        self._ppy = self._h / 2.0

        # Per-state processors — each interprets its model's output its own way.
        self._procs = {
            'WEAPON_CLUB': WeaponProcessor(
                self, 'WEAPON_CLUB', '/vision/task1_target'),
            'MEIHUA_FOREST_EXECUTION': SimpleBestProcessor(
                self, 'MEIHUA_FOREST_EXECUTION', '/vision/task2_target',
                'target_meihua'),
            'MARTIAL_ART_PLACEMENT': GridProcessor(
                self, 'MARTIAL_ART_PLACEMENT', '/vision/task3_target'),
        }

        # Subscribers
        self.create_subscription(String, '/system_status',
                                 self._on_status, RT_QOS, callback_group=self._cb)
        self.create_subscription(String, '/weapon_selection',
                                 self._on_weapon_sel, RELIABLE_QOS, callback_group=self._cb)

        # RealSense
        self._pipeline = rs.pipeline()
        self._align    = None
        self._init_camera()

        # Threads
        threading.Thread(target=self._cam_loop,   daemon=True, name='cam').start()
        threading.Thread(target=self._infer_loop, daemon=True, name='infer').start()

        self.get_logger().info(
            f'VisionNode ready | {self._w}x{self._h}@{self._fps}fps'
            f' | rotate={self._rotate}° | device={self.get_parameter("device").value}'
            f' | YOLO {YOLO_FPS}fps')

    # ── Display geometry (after rotation) ───────────────────────────────────────

    @property
    def disp_w(self):
        return self._h if self._rotate in (90, 270) else self._w

    @property
    def disp_h(self):
        return self._w if self._rotate in (90, 270) else self._h

    # ── Camera init ───────────────────────────────────────────────────────────

    def _init_camera(self):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self._w, self._h, rs.format.bgr8, self._fps)
        cfg.enable_stream(rs.stream.depth, self._w, self._h, rs.format.z16,  self._fps)
        self._pipeline.start(cfg)
        self._align = rs.align(rs.stream.color)
        intr = (self._pipeline.get_active_profile()
                .get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        self._fx, self._fy, self._ppx, self._ppy = rotate_intrinsics(
            intr.fx, intr.fy, intr.ppx, intr.ppy, self._w, self._h, self._rotate)
        self.get_logger().info(
            f'Camera OK | rot intrinsics fx={self._fx:.1f} fy={self._fy:.1f} '
            f'cx={self._ppx:.1f} cy={self._ppy:.1f}')

    def intr(self):
        """Rotated camera intrinsics (fx, fy, ppx, ppy) for the display frame."""
        return (self._fx, self._fy, self._ppx, self._ppy)

    # ── Publisher cache ─────────────────────────────────────────────────────────

    def get_publisher(self, topic, msg_type, reliable=False):
        if topic not in self._pubs:
            qos = RELIABLE_QOS if reliable else RT_QOS
            self._pubs[topic] = self.create_publisher(msg_type, topic, qos)
        return self._pubs[topic]

    # ── Subscribers ───────────────────────────────────────────────────────────

    def _on_status(self, msg: String):
        try:
            state = json.loads(msg.data).get('macro_state', 'IDLE')
            with self._lock:
                if state != self._state:
                    self.get_logger().info(f'State → {state}')
                self._state = state
        except Exception as e:
            self.get_logger().warn(f'status: {e}')

    def _on_weapon_sel(self, msg: String):
        try:
            cls = json.loads(msg.data).get('class', 'spearhead')
            with self._lock:
                self._tgt_cls = cls
            self.get_logger().info(f'weapon_selection → {cls}')
        except Exception as e:
            self.get_logger().warn(f'weapon_sel: {e}')

    # ── Camera thread ─────────────────────────────────────────────────────────

    def _cam_loop(self):
        while self._running:
            try:
                aligned = self._align.process(
                    self._pipeline.wait_for_frames(timeout_ms=2000))
                cf = aligned.get_color_frame()
                df = aligned.get_depth_frame()
                if not cf or not df:
                    continue
                color = np.asanyarray(cf.get_data())
                depth = np.asanyarray(df.get_data())
                try:
                    self._q.put_nowait((color, depth))
                except queue.Full:
                    try: self._q.get_nowait()
                    except queue.Empty: pass
                    self._q.put_nowait((color, depth))
            except Exception as e:
                self.get_logger().warn(f'cam: {e}')
                time.sleep(0.1)

    # ── Inference thread ──────────────────────────────────────────────────────

    def _infer_loop(self):
        interval = 1.0 / YOLO_FPS
        last_t   = 0.0

        while self._running:
            try:
                color, depth = self._q.get(timeout=1.0)
            except queue.Empty:
                continue

            # Apply physical camera rotation once, up front.
            color = rotate_frame(color, self._rotate)
            depth = rotate_frame(depth, self._rotate)

            with self._lock:
                state = self._state

            display = color.copy()
            now     = time.time()
            do_yolo = state != 'IDLE' and (now - last_t >= interval)

            model_file = '—'
            if do_yolo:
                last_t = now
                proc = self._procs.get(state)
                model = self._get_model(STATE_MODEL.get(state))
                if proc is not None and model is not None:
                    conf   = self.get_parameter('conf').value
                    imgsz  = self.get_parameter('imgsz').value
                    device = self.get_parameter('device').value
                    results = model.predict(color, imgsz=imgsz, conf=conf,
                                            device=device, verbose=False)
                    proc.process(results, color, depth, display, self.intr())
                    model_file = os.path.basename(
                        self.get_parameter(STATE_MODEL[state]).value)
                else:
                    model_file = 'no model (abstract)'

            # HUD
            cv2.putText(display, f'STATE: {state}  MODEL: {model_file}',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.drawMarker(display, (int(self._ppx), int(self._ppy)),
                           (80, 80, 80), cv2.MARKER_CROSS, 20, 1)
            with self._frame_lk:
                self._frame = display

    # ── Model management ──────────────────────────────────────────────────────

    def _get_model(self, key):
        if not key:
            return None
        with self._model_lk:
            if self._model_key == key and self._model:
                return self._model
            if self._model:
                self.get_logger().info(f'Unload {self._model_key} → load {key}')
                del self._model
                self._model = None
                gc.collect()
            path = self.get_parameter(key).value
            model = load_model(path, log=self.get_logger().info)
            if model is None:
                self.get_logger().warn(f'Model not found / abstract: {path}')
                return None
            self._model     = model
            self._model_key = key
            return self._model

    def destroy_node(self):
        self._running = False
        try: self._pipeline.stop()
        except: pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, node.disp_w, node.disp_h)

    try:
        while node._running:
            with node._frame_lk:
                frame = node._frame
            if frame is not None:
                cv2.imshow(WIN_NAME, frame)
            if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
