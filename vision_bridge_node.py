#!/usr/bin/env python3
"""
R2 Vision Bridge Node — ABU Robocon 2026
Bridges Flask UI ↔ ROS2 State Machine

Publishes  : /mission_plan    (std_msgs/String, JSON)
Publishes  : /emergency_stop  (std_msgs/Bool) True=ABORT, False=CLEAR
Subscribes : /system_status   (std_msgs/String, JSON)
"""

import json
import math
import time
import threading
import requests

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool

# ── Flask UI endpoint (same machine) ──────────────────────────────────────────
FLASK_BASE = "http://localhost:5000"
STATUS_PUSH_RATE = 0.1   # seconds — 10 Hz to Flask

# ── QoS ───────────────────────────────────────────────────────────────────────
RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
REALTIME_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# ── macro_state values from state machine ─────────────────────────────────────
MACRO_STATES = {'IDLE', 'WEAPON_CLUB', 'MEIHUA_FOREST_EXECUTION', 'MARTIAL_ART_PLACEMENT'}


def sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats so json.dumps never throws."""
    if isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


class UIBridgeNode(Node):

    def __init__(self):
        super().__init__('r2_vision_bridge')

        self._cb = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self._last_mission_time = 0.0  # debounce: prevent double START publish

        # Latest status received from state machine
        self._robot_status = {
            'macro_state': 'IDLE',
            'imu_pitch': 0.0,
            'hardware_error': 'None',
        }

        # ── Publishers ────────────────────────────────────────────────────────
        self._mission_pub = self.create_publisher(
            String, '/mission_plan', RELIABLE_QOS
        )
        self._estop_pub = self.create_publisher(
            Bool, '/emergency_stop', RELIABLE_QOS
        )

        # ── Subscriber ────────────────────────────────────────────────────────
        self.create_subscription(
            String,
            '/system_status',
            self._system_status_cb,
            REALTIME_QOS,
            callback_group=self._cb,
        )

        # ── Timers ────────────────────────────────────────────────────────────
        # Poll Flask for pending mission command, push status back to Flask
        self.create_timer(STATUS_PUSH_RATE, self._sync_with_flask,
                          callback_group=self._cb)

        self.get_logger().info('UI Bridge ready — connected to Flask at ' + FLASK_BASE)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _system_status_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            if data.get('macro_state') not in MACRO_STATES:
                return
            with self._lock:
                self._robot_status.update(data)
        except (json.JSONDecodeError, Exception) as e:
            self.get_logger().error(f'[system_status_cb] {e}')

    def _sync_with_flask(self):
        """10 Hz: push robot status → Flask, pull pending command → ROS2."""
        try:
            # 1. Push robot status to Flask UI
            with self._lock:
                status_snapshot = dict(self._robot_status)

            requests.post(
                f'{FLASK_BASE}/api/robot_status',
                json=sanitize_for_json(status_snapshot),
                timeout=0.08,
            )

            # 2. Pull pending mission command from Flask UI
            res = requests.get(f'{FLASK_BASE}/api/control', timeout=0.08)
            data = res.json()

            cmd = data.get('latest_command')
            if cmd == 'START':
                now = time.monotonic()
                if now - self._last_mission_time > 2.0:  # 2s debounce
                    self._last_mission_time = now
                    self._send_mission_plan(data)
                    requests.post(
                        f'{FLASK_BASE}/api/control/action',
                        json={'action': 'None'},
                        timeout=0.08,
                    )
            elif cmd == 'DISENGAGE':
                self._send_estop(True)
                requests.post(
                    f'{FLASK_BASE}/api/control/action',
                    json={'action': 'None'},
                    timeout=0.08,
                )
            elif cmd == 'RESET':
                self._send_estop(False)
                requests.post(
                    f'{FLASK_BASE}/api/control/action',
                    json={'action': 'None'},
                    timeout=0.08,
                )

        except requests.exceptions.ConnectionError:
            pass  # Flask not running yet — silently skip
        except Exception as e:
            self.get_logger().debug(f'[sync_with_flask] {e}')

    # ── Emergency stop ────────────────────────────────────────────────────────

    def _send_estop(self, active: bool):
        msg = Bool()
        msg.data = active
        self._estop_pub.publish(msg)
        self.get_logger().warn(f'[emergency_stop] {"ABORT" if active else "CLEAR"}')

    # ── Mission publish ────────────────────────────────────────────────────────

    def _send_mission_plan(self, flask_data: dict):
        """Build mission JSON from Flask state and publish to /mission_plan."""
        try:
            plan = {
                'command': 'start_auto',
                'team_color': flask_data.get('team_color', 'BLUE'),
                'yolo_conf': float(flask_data.get('yolo_conf', 0.75)),
                'weapon_club_setup': {
                    'head_index': int(flask_data.get('selected_spearhead', 1)),
                },
                'meihua_forest_path': flask_data.get('meihua_forest_path', []),
                'martial_art_zone': {
                    'place_target': flask_data.get('ttt_target', 'middle'),
                },
            }
            msg = String()
            msg.data = json.dumps(plan)
            self._mission_pub.publish(msg)
            self.get_logger().info(
                f'[mission_plan] published — spearhead={plan["weapon_club_setup"]["head_index"]}'
            )
        except Exception as e:
            self.get_logger().error(f'[send_mission_plan] {e}')

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        self.get_logger().info('UI Bridge shutting down.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UIBridgeNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
