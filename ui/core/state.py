import threading

state_lock = threading.Lock()

robot_state = {
    "x": 0.0, "y": 0.0, "theta": 0.0,
    "task1_step": 1,
    "selected_spearhead": 1,
    "latest_command": "None",
    "logs": ["> System Initialized.", "> Waiting for start command..."],
    "team_color": "BLUE",
    "meihua_forest_path": [],
    "ttt_target": "middle",
    "yolo_conf": 0.75,
    "macro_state": "IDLE",
    "imu_pitch": 0.0,
    "hardware_error": "None",
}
