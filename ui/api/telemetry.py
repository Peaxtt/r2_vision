"""
Flask camera telemetry.

The live video is NOT served by Flask — yolo_node.py streams its annotated
frames over MJPEG on http://<host>:8080/video_feed, and the UI <img> points
straight at it (see index.html streamURL()). This keeps a single camera owner
(yolo_node) and works for webcam/RealSense alike. These endpoints are just
small status/diagnostic stubs.
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
