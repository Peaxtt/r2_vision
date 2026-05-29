# R2 Vision — ABU Robocon 2026

## Structure
```
r2_vision/
├── camera_test.py   — ทดสอบกล้องโดดๆ (ไม่ต้องมี ROS)
├── yolo_node.py     — ROS2 node หลัก (กล้อง + YOLO + publish)
└── models/
    ├── weapon.onnx  — Task 1 spearhead detection
    ├── cube.pt      — Task 2 cube detection
    └── martial.pt   — Task 3 (ยังไม่มี)
```

## วิธีใช้

### เทสกล้อง (ไม่ต้อง ROS)
```bash
python3 camera_test.py
```

### รัน YOLO node
```bash
python3 yolo_node.py
```

### Switch model ผ่าน topic
```bash
# WEAPON_CLUB
ros2 topic pub /system_status std_msgs/msg/String \
  '{data: "{\"macro_state\": \"WEAPON_CLUB\", \"imu_pitch\": 0.0, \"hardware_error\": \"None\"}"}'

# MEIHUA
ros2 topic pub /system_status std_msgs/msg/String \
  '{data: "{\"macro_state\": \"MEIHUA_FOREST_EXECUTION\", \"imu_pitch\": 0.0, \"hardware_error\": \"None\"}"}'

# Fallback spearhead → fist
ros2 topic pub /weapon_selection std_msgs/msg/String \
  '{data: "{\"class\": \"fist\", \"index\": 1}"}'
```

## Model paths
แก้ใน `yolo_node.py` ส่วน MODEL_WEAPON / MODEL_MEIHUA / MODEL_MARTIAL
