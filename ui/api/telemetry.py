"""
Flask camera telemetry — video feed removed.
Camera display is handled by yolo_detection_node OpenCV window.
"""
import threading
from flask import Blueprint, jsonify

telemetry_bp = Blueprint("telemetry", __name__)

_frame_count = 0


@telemetry_bp.route("/api/camera/fps")
def camera_fps():
    return jsonify({"frame_count": _frame_count})


@telemetry_bp.route("/api/camera/reset", methods=["POST"])
def camera_reset():
    return jsonify({"status": "ok", "note": "camera owned by yolo_detection_node"})
