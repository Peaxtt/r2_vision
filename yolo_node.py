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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None   # laptop / webcam-only machine without RealSense SDK

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

YOLO_FPS    = 5    # YOLO inference rate cap
CAM_W       = 848
CAM_H       = 480
CAM_FPS     = 30
WIN_NAME    = 'R2 Vision — Q to quit'
STREAM_FPS  = 15   # MJPEG stream rate to the Flask UI
STREAM_PORT = 8080 # MJPEG stream port (UI <img> points here)


# ── MJPEG stream server (serves the annotated display to the web UI) ──────────

def make_mjpeg_server(node, host, port):
    """Tiny multipart/x-mixed-replace server that streams node._frame as JPEG.

    The same annotated frame shown in the OpenCV window (detections + HUD) is
    served to the Flask UI, so the UI shows whatever the node sees — including
    webcam mode, since it is just node._frame.
    """
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # silence per-request logging

        def do_GET(self):
            if self.path.split('?')[0] not in ('/', '/video_feed', '/stream'):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            interval = 1.0 / STREAM_FPS
            try:
                while node._running:
                    with node._frame_lk:
                        frame = node._frame
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    ok, jpg = cv2.imencode(
                        '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not ok:
                        continue
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpg)}\r\n\r\n'.encode())
                    self.wfile.write(jpg.tobytes())
                    self.wfile.write(b'\r\n')
                    time.sleep(interval)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client (browser tab) closed — normal

    return ThreadingHTTPServer((host, port), Handler)


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
        # Camera source: 'realsense' (RGB-D), 'webcam'/'0'/'1'.. (laptop cam),
        # or a path to a video file. webcam/video have NO depth → z/mm become
        # null; pixel-error, grid occupancy and overlays still work.
        self.declare_parameter('camera',         'realsense')
        # Force a macro_state without /system_status (laptop/webcam testing).
        # '' = follow ROS topic as normal. e.g. MEIHUA_FOREST_EXECUTION.
        self.declare_parameter('force_state',     '')
        # MJPEG stream → Flask UI (annotated frame). The UI <img> reads
        # http://<host>:<stream_port>/video_feed.
        self.declare_parameter('stream',          True)
        self.declare_parameter('stream_port',     STREAM_PORT)
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
        self._frame       = None
        self._frame_lk    = threading.Lock()
        self._model_label = '—'    # persists between inference frames (no blink)

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

        # Camera (RealSense or OpenCV webcam/video)
        self._pipeline = None
        self._align    = None
        self._cap      = None      # cv2.VideoCapture for webcam/video mode
        self._has_depth = True
        self._init_camera()

        # Threads
        threading.Thread(target=self._cam_loop,   daemon=True, name='cam').start()
        threading.Thread(target=self._infer_loop, daemon=True, name='infer').start()

        # MJPEG stream server for the web UI
        self._mjpeg = None
        if self.get_parameter('stream').value:
            port = int(self.get_parameter('stream_port').value)
            try:
                self._mjpeg = make_mjpeg_server(self, '0.0.0.0', port)
                threading.Thread(target=self._mjpeg.serve_forever,
                                 daemon=True, name='mjpeg').start()
                self.get_logger().info(
                    f'MJPEG stream ready → http://<this-host>:{port}/video_feed')
            except OSError as e:
                self.get_logger().warn(f'MJPEG stream disabled ({port} in use?): {e}')

        depth_str = 'RGB-D' if self._has_depth else 'RGB-only (no depth)'
        self.get_logger().info(
            f'VisionNode ready | {self._w}x{self._h}@{self._fps}fps {depth_str}'
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
        src = str(self.get_parameter('camera').value).strip()
        if src.lower() in ('', 'realsense', 'rs', 'd435', 'd455'):
            self._init_realsense()
        else:
            self._init_opencv(src)

    def _init_realsense(self):
        if rs is None:
            raise RuntimeError(
                "camera='realsense' but pyrealsense2 is not installed. "
                "Use -p camera:=webcam (or a device index) on a laptop.")
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self._w, self._h, rs.format.bgr8, self._fps)
        cfg.enable_stream(rs.stream.depth, self._w, self._h, rs.format.z16,  self._fps)
        self._pipeline.start(cfg)
        self._align = rs.align(rs.stream.color)
        intr = (self._pipeline.get_active_profile()
                .get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        self._has_depth = True
        self._fx, self._fy, self._ppx, self._ppy = rotate_intrinsics(
            intr.fx, intr.fy, intr.ppx, intr.ppy, self._w, self._h, self._rotate)
        self.get_logger().info(
            f'RealSense OK | rot intrinsics fx={self._fx:.1f} fy={self._fy:.1f} '
            f'cx={self._ppx:.1f} cy={self._ppy:.1f}')

    def _init_opencv(self, src):
        """Open a laptop webcam (index) or a video file. No depth available."""
        target = int(src) if src.isdigit() else (0 if src.lower() == 'webcam' else src)
        self._cap = cv2.VideoCapture(target)
        if isinstance(target, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
            self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera source: {src!r}")

        # Use the actual frame size the device/file gives us.
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or self._w
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self._h
        self._has_depth = False

        # No real intrinsics — approximate from a ~60° horizontal FOV so the
        # principal point (used for pixel-error) is the image centre. mm/z stay
        # null because depth is unavailable.
        fx = fy = self._w / (2.0 * np.tan(np.radians(60.0) / 2.0))
        self._fx, self._fy, self._ppx, self._ppy = rotate_intrinsics(
            fx, fy, self._w / 2.0, self._h / 2.0, self._w, self._h, self._rotate)
        self.get_logger().warn(
            f'OpenCV camera ({src}) OK | {self._w}x{self._h} | NO DEPTH '
            f'(z/mm will be null; pixel-error & grid still work)')

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
        if self._cap is not None:
            self._cam_loop_opencv()
        else:
            self._cam_loop_realsense()

    def _cam_loop_realsense(self):
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
                self._push(color, depth)
            except Exception as e:
                self.get_logger().warn(f'cam: {e}')
                time.sleep(0.1)

    def _cam_loop_opencv(self):
        # Zero depth map (uint16): depth lookups find no valid pixels → None.
        zero_depth = np.zeros((self._h, self._w), dtype=np.uint16)
        delay = 1.0 / max(1, self._fps)
        while self._running:
            ok, color = self._cap.read()
            if not ok:
                # End of video file → loop; webcam hiccup → brief wait.
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue
            self._push(color, zero_depth)
            time.sleep(delay)   # pace to ~cam_fps (webcam read is blocking anyway)

    def _push(self, color, depth):
        try:
            self._q.put_nowait((color, depth))
        except queue.Full:
            try: self._q.get_nowait()
            except queue.Empty: pass
            self._q.put_nowait((color, depth))

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
            forced = self.get_parameter('force_state').value
            if forced:
                state = forced

            now     = time.time()
            do_yolo = state != 'IDLE' and (now - last_t >= interval)

            if do_yolo:
                # Inference frame: annotate fresh and publish it as the display.
                last_t  = now
                display = color.copy()
                task    = self._resolve_state(state)   # WEAPON_CLUB_SETUP → WEAPON_CLUB
                proc    = self._procs.get(task)
                model   = self._get_model(STATE_MODEL.get(task))
                if proc is not None and model is not None:
                    conf   = self.get_parameter('conf').value
                    imgsz  = self.get_parameter('imgsz').value
                    device = self.get_parameter('device').value
                    results = model.predict(color, imgsz=imgsz, conf=conf,
                                            device=device, verbose=False)
                    proc.process(results, color, depth, display, self.intr())
                    self._model_label = os.path.basename(
                        self.get_parameter(STATE_MODEL[task]).value)
                else:
                    self._model_label = 'no model (abstract)'
                self._draw_hud(display, state)
                with self._frame_lk:
                    self._frame = display
            elif state == 'IDLE':
                # Idle: stream live video (no detections to preserve).
                self._model_label = '—'
                display = color.copy()
                self._draw_hud(display, state)
                with self._frame_lk:
                    self._frame = display
            # else: between inferences in an active state → keep the last
            # annotated frame so boxes + label stay on screen (no flicker).

    def _resolve_state(self, state):
        """Map a macro_state to a known task key.

        Exact match wins; otherwise a prefix match, so finer sub-states from
        the state machine (e.g. WEAPON_CLUB_SETUP, WEAPON_CLUB_AIM) still drive
        the right task instead of falling through to 'no model (abstract)'.
        """
        if state in self._procs:
            return state
        for key in self._procs:
            if state.startswith(key):
                return key
        return state

    def _draw_hud(self, display, state):
        cv2.putText(display, f'STATE: {state}  MODEL: {self._model_label}',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.drawMarker(display, (int(self._ppx), int(self._ppy)),
                       (80, 80, 80), cv2.MARKER_CROSS, 20, 1)

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
        try:
            if self._mjpeg is not None:
                self._mjpeg.shutdown()
        except Exception:
            pass
        try:
            if self._pipeline is not None:
                self._pipeline.stop()
        except Exception:
            pass
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
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
