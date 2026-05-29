import logging
from flask import Blueprint, request, jsonify
from core.state import robot_state, state_lock

control_bp = Blueprint("control", __name__)
log = logging.getLogger("abu.control")


@control_bp.route("/api/telemetry", methods=["POST"])
def api_telemetry():
    data = request.json or {}
    with state_lock:
        robot_state["x"] = float(data.get("x", 0.0))
        robot_state["y"] = float(data.get("y", 0.0))
        robot_state["theta"] = float(data.get("theta", 0.0))
    return jsonify({"status": "ok"})


@control_bp.route("/api/logs", methods=["POST", "GET"])
def api_logs():
    if request.method == "POST":
        data = request.json or {}
        msg = data.get("message", "")
        if msg:
            with state_lock:
                robot_state["logs"].append(msg)
                if len(robot_state["logs"]) > 100:
                    robot_state["logs"].pop(0)
        return jsonify({"status": "ok"})
    else:
        with state_lock:
            return jsonify(robot_state["logs"])


@control_bp.route("/api/task1/step", methods=["POST"])
def api_task1_step():
    data = request.json or {}
    with state_lock:
        robot_state["task1_step"] = int(data.get("step", 1))
    return jsonify({"status": "ok"})


@control_bp.route("/api/control/action", methods=["POST"])
def api_control_action():
    data = request.json or {}
    action = data.get("action", "None")
    msg = None

    with state_lock:
        # ── Config setters (no latest_command change) ──────────────────────
        if action == "SET_TEAM_COLOR":
            color = data.get("color", "BLUE").upper()
            if color in ("RED", "BLUE"):
                robot_state["team_color"] = color
            msg = f"> Config: Team Color → {robot_state['team_color']}"

        elif action == "SET_MEIHUA_PATH":
            robot_state["meihua_forest_path"] = data.get("path", [])
            n = len(robot_state["meihua_forest_path"])
            msg = f"> Config: Meihua Path confirmed ({n} waypoints)"

        elif action == "SET_TTT_SLOT":
            slot_map = {1: "left", 2: "middle", 3: "right"}
            slot = int(data.get("slot", 2))
            robot_state["ttt_target"] = slot_map.get(slot, "middle")
            msg = f"> Config: Tic-Tac-Toe Target → {robot_state['ttt_target'].upper()}"

        elif action == "SET_YOLO_CONF":
            robot_state["yolo_conf"] = float(data.get("value", 0.75))
            msg = f"> Config: YOLO Confidence → {robot_state['yolo_conf']:.2f}"

        # ── Operator commands ──────────────────────────────────────────────
        else:
            robot_state["latest_command"] = action
            if "target_spearhead" in data:
                robot_state["selected_spearhead"] = int(data["target_spearhead"])

            if action == "START":
                msg = (
                    f"> START: color={robot_state['team_color']} "
                    f"spearhead={robot_state['selected_spearhead']} "
                    f"path={robot_state['meihua_forest_path']} "
                    f"ttt={robot_state['ttt_target']}"
                )
            elif action == "DISENGAGE":
                msg = "> EMERGENCY DISENGAGE"
            elif action == "RESET":
                msg = "> FIELD RESET"
            elif action != "None":
                msg = f"> Command: {action}"

        if msg:
            log.info(msg.lstrip("> "))
            robot_state["logs"].append(msg)
            if len(robot_state["logs"]) > 100:
                robot_state["logs"].pop(0)

    return jsonify({"status": "ok"})


@control_bp.route("/api/control", methods=["GET"])
def api_control_get():
    with state_lock:
        return jsonify({
            "latest_command":     robot_state["latest_command"],
            "selected_spearhead": robot_state["selected_spearhead"],
            "task1_step":         robot_state["task1_step"],
            "team_color":         robot_state["team_color"],
            "meihua_forest_path": robot_state["meihua_forest_path"],
            "ttt_target":         robot_state["ttt_target"],
            "yolo_conf":          robot_state["yolo_conf"],
        })


@control_bp.route("/api/status", methods=["GET"])
def api_status():
    with state_lock:
        return jsonify({
            "x": robot_state["x"],
            "y": robot_state["y"],
            "theta": robot_state["theta"],
            "task1_step": robot_state["task1_step"],
            "selected_spearhead": robot_state["selected_spearhead"],
            "latest_command": robot_state["latest_command"],
            "macro_state": robot_state["macro_state"],
            "imu_pitch": robot_state["imu_pitch"],
            "hardware_error": robot_state["hardware_error"],
        })


@control_bp.route("/api/robot_status", methods=["POST"])
def api_robot_status():
    """Receives macro_state + hardware status from ROS2 bridge node."""
    data = request.json or {}
    with state_lock:
        robot_state["macro_state"] = data.get("macro_state", robot_state["macro_state"])
        robot_state["imu_pitch"] = float(data.get("imu_pitch", 0.0))
        robot_state["hardware_error"] = data.get("hardware_error", "None")
    return jsonify({"status": "ok"})
