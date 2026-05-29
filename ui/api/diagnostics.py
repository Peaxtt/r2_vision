import subprocess
from flask import Blueprint, request, jsonify

diagnostics_bp = Blueprint("diagnostics", __name__)

_vision_state = {"macro_state": "IDLE"}

@diagnostics_bp.route("/api/test/vision_state", methods=["POST", "GET"])
def api_vision_state():
    """Set macro_state for testing YOLO node model switching."""
    if request.method == "GET":
        return jsonify(_vision_state)
    data = request.json or {}
    state = data.get("state", "IDLE")
    _vision_state["macro_state"] = state
    payload = f'{{"macro_state": "{state}", "imu_pitch": 0.0, "hardware_error": "None"}}'
    try:
        subprocess.Popen([
            "ros2", "topic", "pub", "--once",
            "/system_status", "std_msgs/msg/String",
            f'{{data: "{payload.replace(chr(34), chr(92)+chr(34))}"}}'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "ok", "state": state})


@diagnostics_bp.route("/api/test/motor", methods=["POST"])
def api_test_motor():
    data = request.json or {}
    motor_id = data.get("motor_id")
    speed = data.get("speed", 0)
    # Stub — wire to ROS2 bridge when motor driver is available
    return jsonify({"status": "ok", "motor_id": motor_id, "speed": speed, "rpm": int(abs(speed) * 12)})


@diagnostics_bp.route("/api/test/gripper", methods=["POST"])
def api_test_gripper():
    data = request.json or {}
    action = data.get("action", "OPEN")
    force = 45.2 if action == "CLOSE" else 0.0
    return jsonify({"status": "ok", "action": action, "force": force})


@diagnostics_bp.route("/api/test/encoder/reset", methods=["POST"])
def api_test_encoder_reset():
    return jsonify({"status": "ok"})


@diagnostics_bp.route("/api/test/camera/config", methods=["POST"])
def api_test_camera_config():
    """Apply brightness + exposure to the live RealSense pipeline."""
    data = request.json or {}
    # slider 0-100 → RS brightness range -64..64
    brightness_slider = int(data.get("brightness", 50))
    exposure_slider   = float(data.get("exposure", 1.0))   # 0.1–10.0

    pipeline = telemetry._pipeline
    if pipeline is None:
        return jsonify({"status": "error", "message": "Camera not started"}), 503

    try:
        import pyrealsense2 as rs
        profile      = pipeline.get_active_profile()
        color_sensor = profile.get_device().first_color_sensor()
        applied      = {}

        # brightness: slider 0-100 → RS -64..64
        if color_sensor.supports(rs.option.brightness):
            rs_bright = int((brightness_slider - 50) * 64 / 50)  # 0→-64, 50→0, 100→64
            rs_bright = max(-64, min(64, rs_bright))
            color_sensor.set_option(rs.option.brightness, rs_bright)
            applied["brightness_rs"] = rs_bright

        # exposure: only adjust if auto-exposure is already OFF
        # (don't force-disable auto — that's what caused white screen)
        if color_sensor.supports(rs.option.enable_auto_exposure):
            auto_on = color_sensor.get_option(rs.option.enable_auto_exposure)
            if not auto_on and color_sensor.supports(rs.option.exposure):
                r = color_sensor.get_option_range(rs.option.exposure)
                exposure_us = r.min + (exposure_slider / 10.0) * (r.max - r.min)
                exposure_us = max(r.min, min(r.max, exposure_us))
                color_sensor.set_option(rs.option.exposure, exposure_us)
                applied["exposure_us"] = exposure_us
            else:
                applied["exposure"] = "auto (unchanged)"

        return jsonify({"status": "ok", "applied": applied})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@diagnostics_bp.route("/api/test/camera/auto_exposure", methods=["POST"])
def api_camera_auto_exposure():
    """Toggle auto-exposure on/off."""
    data    = request.json or {}
    enable  = bool(data.get("enable", True))
    pipeline = telemetry._pipeline
    if pipeline is None:
        return jsonify({"status": "error", "message": "Camera not started"}), 503
    try:
        import pyrealsense2 as rs
        sensor = pipeline.get_active_profile().get_device().first_color_sensor()
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 1 if enable else 0)
        return jsonify({"status": "ok", "auto_exposure": enable})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@diagnostics_bp.route("/api/test/camera/capture", methods=["POST"])
def api_test_camera_capture():
    """Save current frame to disk."""
    import cv2, numpy as np, time

    pipeline = telemetry._pipeline
    if pipeline is None:
        return jsonify({"status": "error", "message": "Camera not started"}), 503

    try:
        frames      = pipeline.wait_for_frames(timeout_ms=2000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return jsonify({"status": "error", "message": "No frame"}), 500

        img  = np.asanyarray(color_frame.get_data())
        path = f"/tmp/frame_capture_{int(time.time())}.png"
        cv2.imwrite(path, img)
        return jsonify({"status": "ok", "file": path})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
