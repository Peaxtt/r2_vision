#!/usr/bin/env python3
"""
YOLO Detection ROS2 Node — ABU Robocon 2026 R2
กล้อง + YOLO + OpenCV window + ROS2 publisher

Publishes:
  /vision/task1_target  PoseStamped  BEST_EFFORT  pixel error + depth (mm)
  /vision/task2_target  PoseStamped  BEST_EFFORT
  /vision/task3_target  PoseStamped  BEST_EFFORT

Subscribes:
  /system_status    String JSON  BEST_EFFORT  macro_state → switch model
  /weapon_selection String JSON  RELIABLE     runtime class switch (fallback)

Display: OpenCV window บน main thread (ไม่ค้าง)
Model:   ONE model loaded at a time (unload เก่าก่อน load ใหม่)
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
from geometry_msgs.msg import PoseStamped

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# ── QoS ───────────────────────────────────────────────────────────────────────
RT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST, depth=1)
RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST, depth=10)

# ── State → model/topic mapping ───────────────────────────────────────────────
STATE_MODEL = {
    'WEAPON_CLUB':             'model_weapon',
    'MEIHUA_FOREST_EXECUTION': 'model_meihua',
    'MARTIAL_ART_PLACEMENT':   'model_martial',
}
STATE_TOPIC = {
    'WEAPON_CLUB':             '/vision/task1_target',
    'MEIHUA_FOREST_EXECUTION': '/vision/task2_target',
    'MARTIAL_ART_PLACEMENT':   '/vision/task3_target',
}

# ── Centroid offset ต่อ class (cx%, cy% ของ bbox) ────────────────────────────
CENTROID = {
    'spearhead':  (0.532, 0.892),
    'fist':       (0.493, 0.734),
    'hand':       (0.330, 0.631),
    'blue_cube':  (0.500, 0.500),
    'red_cube':   (0.500, 0.500),
}

COLORS = {
    'spearhead':  (0,   165, 255),
    'fist':       (0,   255,   0),
    'hand':       (255,   0,   0),
    'blue_cube':  (255, 100,   0),
    'red_cube':   (0,    50, 255),
}

# ── Model paths (เปลี่ยนตรงนี้เมื่อ pull model ใหม่) ─────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_WEAPON  = os.path.join(MODELS_DIR, 'weapon.onnx')
MODEL_MEIHUA  = os.path.join(MODELS_DIR, 'cube.pt')
MODEL_MARTIAL = os.path.join(MODELS_DIR, 'martial.pt')

YOLO_FPS   = 5    # YOLO inference rate cap
CAM_W      = 848
CAM_H      = 480
CAM_FPS    = 30
WIN_NAME   = 'R2 Vision — Q to quit'


class VisionNode(Node):

    def __init__(self):
        super().__init__('r2_vision_node')
        self._cb = ReentrantCallbackGroup()

        # Parameters (override via --ros-args -p)
        self.declare_parameter('model_weapon',  MODEL_WEAPON)
        self.declare_parameter('model_meihua',  MODEL_MEIHUA)
        self.declare_parameter('model_martial', MODEL_MARTIAL)
        self.declare_parameter('conf',           0.50)
        self.declare_parameter('target_weapon',  'spearhead')
        self.declare_parameter('target_meihua',  '')      # '' = ทุก class
        self.declare_parameter('cam_width',      CAM_W)
        self.declare_parameter('cam_height',     CAM_H)
        self.declare_parameter('cam_fps',        CAM_FPS)
        self.declare_parameter('max_depth_mm',   3400)

        # State
        self._state   = 'IDLE'
        self._tgt_cls = self.get_parameter('target_weapon').value
        self._lock    = threading.Lock()
        self._running = True

        # Camera queue
        self._q: queue.Queue = queue.Queue(maxsize=1)

        # Display buffer (main thread reads this)
        self._frame     = None
        self._frame_lk  = threading.Lock()

        # ONE active model
        self._model     = None
        self._model_key = ''
        self._model_lk  = threading.Lock()

        # Publishers cache
        self._pubs: dict = {}

        # Camera params
        self._w   = self.get_parameter('cam_width').value
        self._h   = self.get_parameter('cam_height').value
        self._fps = self.get_parameter('cam_fps').value
        self._ppx = self._w / 2.0
        self._ppy = self._h / 2.0

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
            f' | YOLO {YOLO_FPS}fps | models dir: {MODELS_DIR}')

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
        self._ppx, self._ppy = intr.ppx, intr.ppy
        self.get_logger().info(
            f'Camera OK | principal=({self._ppx:.1f},{self._ppy:.1f})')

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

            with self._lock:
                state   = self._state
                tgt_cls = self._tgt_cls

            display = color.copy()
            now     = time.time()
            do_yolo = state != 'IDLE' and (now - last_t >= interval)

            if do_yolo:
                last_t = now
                model  = self._get_model(STATE_MODEL.get(state))
                if model:
                    conf    = self.get_parameter('conf').value
                    results = model(color, conf=conf, verbose=False)
                    tc      = tgt_cls if state == 'WEAPON_CLUB' else \
                              (self.get_parameter('target_meihua').value or None)
                    best    = self._pick_best(results, tc)

                    if best:
                        x1, y1, x2, y2, cls_name = best
                        cx_p, cy_p = CENTROID.get(cls_name, (0.5, 0.5))
                        scx = int(np.clip(x1 + (x2-x1)*cx_p, 0, self._w-1))
                        scy = int(np.clip(y1 + (y2-y1)*cy_p, 0, self._h-1))
                        pad = depth[max(0,scy-3):min(self._h,scy+4),
                                    max(0,scx-3):min(self._w,scx+4)]
                        md  = self.get_parameter('max_depth_mm').value
                        val = pad[(pad > 0) & (pad <= md)]
                        d_mm = float(np.median(val)) if val.size > 0 else 0.0
                        x_err = float(scx - self._ppx)
                        y_err = float(scy - self._ppy)
                        self._pub(state, x_err, y_err, d_mm)
                        col = COLORS.get(cls_name, (0, 255, 0))
                        cv2.rectangle(display, (x1, y1), (x2, y2), col, 2)
                        cv2.drawMarker(display, (scx, scy),
                                       (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
                        cv2.putText(display,
                                    f'{cls_name} x={x_err:.0f}px z={d_mm:.0f}mm',
                                    (x1, max(y1-8, 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            # HUD
            model_file = os.path.basename(
                self.get_parameter(STATE_MODEL.get(state,'model_weapon')).value
            ) if state != 'IDLE' else '—'
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
            if not path or not os.path.exists(path):
                self.get_logger().warn(f'Model not found: {path}')
                return None
            self.get_logger().info(f'Loading {os.path.basename(path)}')
            self._model     = YOLO(path)
            self._model_key = key
            return self._model

    # ── Pick best detection ───────────────────────────────────────────────────

    def _pick_best(self, results, target_cls):
        best_conf, best_box = -1.0, None
        for r in results:
            if not r.boxes:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                name = r.names.get(int(box.cls[0]), '')
                if target_cls and name != target_cls:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    x = box.xyxy[0].cpu().numpy().astype(int)
                    best_box = (x[0], x[1], x[2], x[3], name)
        return best_box

    # ── Publisher ─────────────────────────────────────────────────────────────

    def _pub(self, state, x_err, y_err, depth_mm):
        topic = STATE_TOPIC.get(state)
        if not topic:
            return
        if topic not in self._pubs:
            self._pubs[topic] = self.create_publisher(PoseStamped, topic, RT_QOS)
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.pose.position.x = x_err
        msg.pose.position.y = y_err
        msg.pose.position.z = depth_mm
        self._pubs[topic].publish(msg)

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
    cv2.resizeWindow(WIN_NAME, node._w, node._h)

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
