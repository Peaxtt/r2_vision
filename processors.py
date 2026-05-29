#!/usr/bin/env python3
"""Per-task detection processors for r2_vision.

Each macro_state runs its OWN processing pipeline behind a common
``TaskProcessor`` interface, so swapping the active model also swaps how its
detections are interpreted, drawn and published:

  • WeaponProcessor  (WEAPON_CLUB)              — full OpenVINO weapon pipeline:
        per-class centroid config, bbox/alignment/depth filters,
        placement-index assignment, mm coordinates from the principal point.
        Publishes the complete detection list (mm + placement index) as JSON.

  • SimpleBestProcessor (MEIHUA_FOREST_EXECUTION) — pick the single best target
        of a class, publish pixel-error + depth PoseStamped (legacy contract).

  • AbstractProcessor (MARTIAL_ART_PLACEMENT / any task with no model) —
        placeholder.  Draws a "no model" HUD and publishes nothing.

A processor is handed the YOLO results for one frame plus the rotated camera
intrinsics; the node owns the camera, model loading and the display window.
"""

import json
import math
from itertools import product as iproduct

import cv2
import numpy as np

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped


# ── Base ────────────────────────────────────────────────────────────────────

class TaskProcessor:
    """Common interface. ``node`` exposes params, logger and a publisher cache."""

    def __init__(self, node, state):
        self.node = node
        self.state = state
        self._configured_for = None   # class-name dict the config was built for

    def log(self, msg):
        self.node.get_logger().info(f'[{self.state}] {msg}')

    def configure(self, class_names):
        """Hook called whenever the active model's class names change."""
        self._configured_for = class_names

    def _ensure_configured(self, class_names):
        if self._configured_for != class_names:
            self.configure(class_names)

    def process(self, results, color, depth, display, intr):
        """Interpret one frame of YOLO results.  Subclasses override.

        intr = (fx, fy, ppx, ppy) for the (rotated) color frame.
        """
        raise NotImplementedError


# ── WEAPON_CLUB: full OpenVINO weapon pipeline ───────────────────────────────

class WeaponProcessor(TaskProcessor):
    """Ports weapon_detection_vino.py into a ROS2 processor.

    Output: a String JSON message on the task topic carrying every detection
    with its placement index and mm coordinates — mirrors the records that the
    standalone script writes to detections.json.
    """

    LAYOUT = ['spearhead', 'fist', 'hand', 'hand', 'fist', 'spearhead']

    # Filter / centroid defaults (overridden by centroid_config.txt).
    DEFAULTS = {
        'max_depth_mm':      5000,
        'max_bbox_area_pct': 50.0,
        'horiz_align_pct':   10.0,
        'seg_depth_dev_pct': 35.0,
        'depth_tol_mm':      150,
    }

    def __init__(self, node, state, topic):
        super().__init__(node, state)
        self.topic = topic
        self.cfg = dict(self.DEFAULTS)
        self.cent_x = {}   # cls_id → percent into bbox
        self.cent_y = {}
        self.colors = {}

    # -- config -------------------------------------------------------------

    def configure(self, class_names):
        super().configure(class_names)
        self.cfg = dict(self.DEFAULTS)
        self.cent_x = {idx: 50.0 for idx in class_names}
        self.cent_y = {idx: 50.0 for idx in class_names}
        self._load_config(class_names)

        rng = np.random.default_rng(42)
        self.colors = {idx: tuple(int(c) for c in rng.integers(80, 255, 3))
                       for idx in class_names}

    def _load_config(self, class_names):
        path = self.node.get_parameter('centroid_config').value
        try:
            data = {}
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        k, v = line.split('=', 1)
                        data[k.strip()] = float(v.strip())

            for key, cast in (('max_depth_mm', int), ('max_bbox_area_pct', float),
                              ('horiz_align_pct', float), ('seg_depth_dev_pct', float),
                              ('depth_tol_mm', int)):
                if key in data:
                    self.cfg[key] = cast(data[key])

            for idx, name in class_names.items():
                if f'{name}_cent_x' in data:
                    self.cent_x[idx] = data[f'{name}_cent_x']
                if f'{name}_cent_y' in data:
                    self.cent_y[idx] = data[f'{name}_cent_y']

            self.log(f'centroid config loaded from {path} | '
                     f'depth_max={self.cfg["max_depth_mm"]}mm '
                     f'bbox_max={self.cfg["max_bbox_area_pct"]:.1f}% '
                     f'depth_tol={self.cfg["depth_tol_mm"]}mm')
        except FileNotFoundError:
            self.log(f'{path} not found — using centroid defaults (50%/50%)')

    # -- filters ------------------------------------------------------------

    def _filter_horiz_align(self, boxes, h_f):
        if len(boxes) <= 1:
            return boxes
        cy_vals = [(b['y1'] + b['y2']) // 2 for b in boxes]
        median_cy = float(np.median(cy_vals))
        max_dev = self.cfg['horiz_align_pct'] / 100.0 * h_f
        return [b for b in boxes
                if abs((b['y1'] + b['y2']) / 2 - median_cy) <= max_dev]

    def _filter_outlier_depth(self, boxes):
        if len(boxes) <= 1:
            return boxes
        z_vals = [b['z_mm'] for b in boxes if b['z_mm']]
        if not z_vals:
            return boxes
        median_z = float(np.median(z_vals))
        thr = self.cfg['seg_depth_dev_pct'] / 100.0
        return [b for b in boxes
                if b['z_mm'] and abs(b['z_mm'] - median_z) / median_z <= thr]

    # -- placement-index assignment -----------------------------------------

    def _assign_placement(self, dets, frame_width):
        n = len(dets)
        if n == 0:
            return {}

        xs = [(d['bbox'][0] + d['bbox'][2]) / 2 for d in dets]
        prefer_right = (sum(xs) / n) > frame_width / 2

        valid_for = []
        for d in dets:
            valid = [i for i, c in enumerate(self.LAYOUT) if c == d['class_name']]
            valid_for.append(valid if valid else [-1])

        best_key, best_combo = None, [v[0] for v in valid_for]
        for combo in iproduct(*valid_for):
            known = [c for c in combo if c != -1]
            if len(set(known)) != len(known):
                continue
            score = 0
            for a in range(n):
                for b in range(a + 1, n):
                    ca, cb = combo[a], combo[b]
                    if ca == -1 or cb == -1:
                        continue
                    score += 1 if (xs[a] < xs[b]) == (ca < cb) else -1
            idx_sum = sum(c for c in combo if c != -1)
            key = (score, idx_sum if prefer_right else -idx_sum)
            if best_key is None or key > best_key:
                best_key, best_combo = key, list(combo)

        return {i: best_combo[i] for i in range(n) if best_combo[i] != -1}

    # -- main ---------------------------------------------------------------

    def process(self, results, color, depth, display, intr):
        fx, fy, ppx, ppy = intr
        h_f, w_f = color.shape[:2]
        frame_area = w_f * h_f

        names = results[0].names if results else {}
        self._ensure_configured(names)

        # Pass 1 — collect, bbox-area filter, depth sampling, mm conversion.
        raw = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_f, x2), min(h_f, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) * (y2 - y1) / frame_area * 100 > self.cfg['max_bbox_area_pct']:
                    continue

                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id))
                cx_pct = self.cent_x.get(cls_id, 50.0)
                cy_pct = self.cent_y.get(cls_id, 50.0)
                scx = max(x1, min(x2 - 1, int(x1 + (x2 - x1) * cx_pct / 100.0)))
                scy = max(y1, min(y2 - 1, int(y1 + (y2 - y1) * cy_pct / 100.0)))

                z_mm = horiz_mm = vert_mm = abs_dist_mm = None
                pad = 3
                patch = depth[max(0, scy - pad):min(h_f, scy + pad + 1),
                              max(0, scx - pad):min(w_f, scx + pad + 1)]
                valid = patch[(patch > 0) & (patch <= self.cfg['max_depth_mm'])]
                if valid.size > 0:
                    obj_d = float(np.median(valid))
                    tol = self.cfg['depth_tol_mm']
                    if tol > 0:
                        band = valid[np.abs(valid.astype(float) - obj_d) <= tol]
                        if band.size > 0:
                            obj_d = float(np.median(band))
                    z_mm = int(obj_d)
                    horiz_mm = round((scx - ppx) * obj_d / fx, 1)
                    vert_mm = round((scy - ppy) * obj_d / fy, 1)
                    abs_dist_mm = round(
                        math.sqrt(horiz_mm ** 2 + vert_mm ** 2 + obj_d ** 2), 1)

                raw.append({
                    'bbox': (x1, y1, x2, y2),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'class_name': cls_name, 'cls_id': cls_id,
                    'conf': float(box.conf[0]),
                    'color': self.colors.get(cls_id, (0, 255, 0)),
                    'centroid_px': (scx, scy),
                    'z_mm': z_mm, 'horiz_mm': horiz_mm,
                    'vert_mm': vert_mm, 'abs_dist_mm': abs_dist_mm,
                })

        # Pass 2 — group filters.
        raw = self._filter_horiz_align(raw, h_f)
        raw = self._filter_outlier_depth(raw)

        # Pass 3 — placement indices.
        placement = self._assign_placement(raw, w_f)
        dets = []
        for i, rb in enumerate(raw):
            rb['index'] = placement.get(i, -1)
            dets.append(rb)

        # Pass 4 — draw + publish.
        self._draw(display, dets, (int(ppx), int(ppy)))
        self._publish(dets)

    def _draw(self, display, dets, ref_px):
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            scx, scy = d['centroid_px']
            col = d['color']
            cv2.rectangle(display, (x1, y1), (x2, y2), col, 2)
            cv2.line(display, (scx, y1), (scx, y2), (180, 255, 180), 1)
            cv2.line(display, (x1, scy), (x2, scy), (180, 255, 180), 1)
            cv2.drawMarker(display, (scx, scy), (0, 255, 255),
                           cv2.MARKER_CROSS, 14, 2)
            if d['z_mm'] is not None:
                cv2.putText(display, f"abs:{d['abs_dist_mm']}mm z:{d['z_mm']}mm",
                            (scx + 8, scy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.36, (0, 255, 255), 1)
            lbl = f"[{d['index']}] {d['class_name']} {d['conf']:.2f}"
            cv2.putText(display, lbl, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.drawMarker(display, ref_px, (100, 180, 100), cv2.MARKER_CROSS, 16, 1)

    def _publish(self, dets):
        records = [{
            'index':               d['index'],
            'class':               d['class_name'],
            'conf':                round(d['conf'], 3),
            'x_from_principle_mm': d['horiz_mm'],
            'y_from_principle_mm': d['vert_mm'],
            'z_depth_mm':          d['z_mm'],
            'abs_dist_mm':         d['abs_dist_mm'],
        } for d in dets]

        pub = self.node.get_publisher(self.topic, String, reliable=False)
        msg = String()
        msg.data = json.dumps({
            'task': self.state,
            'stamp': self.node.get_clock().now().nanoseconds,
            'detections': records,
        })
        pub.publish(msg)


# ── MEIHUA: single-best target → pixel-error PoseStamped (legacy) ────────────

class SimpleBestProcessor(TaskProcessor):
    """Pick the highest-confidence detection of the target class, publish its
    pixel error from image centre plus depth (mm) as a PoseStamped.  This is
    the original yolo_node contract, kept for tasks the state machine already
    consumes that way."""

    CENTROID = {
        'blue_cube': (0.500, 0.500),
        'red_cube':  (0.500, 0.500),
    }
    COLORS = {
        'blue_cube': (255, 100, 0),
        'red_cube':  (0, 50, 255),
    }

    def __init__(self, node, state, topic, target_param):
        super().__init__(node, state)
        self.topic = topic
        self.target_param = target_param

    def _pick_best(self, results, target_cls):
        best_conf, best = -1.0, None
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
                    best = (x[0], x[1], x[2], x[3], name)
        return best

    def process(self, results, color, depth, display, intr):
        _, _, ppx, ppy = intr
        h_f, w_f = color.shape[:2]
        target = self.node.get_parameter(self.target_param).value or None
        best = self._pick_best(results, target)
        if not best:
            return

        x1, y1, x2, y2, cls_name = best
        cx_p, cy_p = self.CENTROID.get(cls_name, (0.5, 0.5))
        scx = int(np.clip(x1 + (x2 - x1) * cx_p, 0, w_f - 1))
        scy = int(np.clip(y1 + (y2 - y1) * cy_p, 0, h_f - 1))

        md = self.node.get_parameter('max_depth_mm').value
        pad = depth[max(0, scy - 3):min(h_f, scy + 4),
                    max(0, scx - 3):min(w_f, scx + 4)]
        val = pad[(pad > 0) & (pad <= md)]
        d_mm = float(np.median(val)) if val.size > 0 else 0.0
        x_err, y_err = float(scx - ppx), float(scy - ppy)

        self._publish(x_err, y_err, d_mm)

        col = self.COLORS.get(cls_name, (0, 255, 0))
        cv2.rectangle(display, (x1, y1), (x2, y2), col, 2)
        cv2.drawMarker(display, (scx, scy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(display, f'{cls_name} x={x_err:.0f}px z={d_mm:.0f}mm',
                    (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    def _publish(self, x_err, y_err, depth_mm):
        pub = self.node.get_publisher(self.topic, PoseStamped, reliable=False)
        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.pose.position.x = x_err
        msg.pose.position.y = y_err
        msg.pose.position.z = depth_mm
        pub.publish(msg)


# ── No-model placeholder ─────────────────────────────────────────────────────

class AbstractProcessor(TaskProcessor):
    """Placeholder for a task whose model does not exist yet.  Never receives
    results (the node skips inference when no model loads); only contributes a
    HUD hint so the operator knows the task is wired but unimplemented."""

    def process(self, results, color, depth, display, intr):
        # No model → nothing to interpret.  Drawing handled by the node HUD.
        pass
