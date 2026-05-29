# R2 Vision — ABU Robocon 2026

Vision system สำหรับหุ่นยนต์ R2 ประกอบด้วย YOLO detection node + ROS2 bridge + Flask UI

---

## โครงสร้างไฟล์

```
r2_vision/
├── camera_test.py          ← ทดสอบกล้องโดดๆ (ไม่ต้องมี ROS)
├── yolo_node.py            ← YOLO detection + ROS2 publisher (ไฟล์หลัก)
├── vision_bridge_node.py   ← เชื่อม Flask UI ↔ ROS2 state machine
└── models/
    ├── weapon.onnx         ← Task 1: spearhead detection
    ├── cube.pt             ← Task 2: cube detection
    └── martial.pt          ← Task 3: (ยังไม่มี)
```

Flask UI (mission control) อยู่ที่: `https://github.com/tawanNophaket/R2-ABU`

---

## ความต้องการ

```bash
pip install pyrealsense2 ultralytics opencv-python numpy
```

ROS2 Humble (สำหรับ yolo_node.py และ vision_bridge_node.py):
```bash
source /opt/ros/humble/setup.bash
```

---

## 1. ทดสอบกล้องโดดๆ

**รันก่อนเสมอ** เพื่อยืนยันว่ากล้องใช้ได้

```bash
python3 camera_test.py
```

- กด `Q` หรือ `ESC` เพื่อออก
- ต้องเห็น FPS > 20 และ depth เปลี่ยนตามระยะจริง
- ถ้ากล้องไม่ขึ้น → ถอดเสียบ USB ใหม่ หรือเช็คว่า process อื่นถือกล้องอยู่

---

## 2. YOLO Detection Node

ไฟล์หลัก — เปิดกล้อง, รัน YOLO, แสดง OpenCV window, publish ตำแหน่งไปยัง ROS2

### รัน
```bash
source /opt/ros/humble/setup.bash
python3 yolo_node.py
```

### ปรับค่าผ่าน ROS2 arguments
```bash
python3 yolo_node.py --ros-args \
  -p model_weapon:=/path/to/weapon.onnx \
  -p conf:=0.5
```

### Model paths
แก้ใน `yolo_node.py` บรรทัดนี้:
```python
MODEL_WEAPON  = os.path.join(MODELS_DIR, 'weapon.onnx')
MODEL_MEIHUA  = os.path.join(MODELS_DIR, 'cube.pt')
MODEL_MARTIAL = os.path.join(MODELS_DIR, 'martial.pt')
```

### Topics ที่ publish

| Topic | Type | เมื่อไร |
|-------|------|---------|
| `/vision/task1_target` | PoseStamped | WEAPON_CLUB + detect spearhead |
| `/vision/task2_target` | PoseStamped | MEIHUA_FOREST_EXECUTION + detect cube |
| `/vision/task3_target` | PoseStamped | MARTIAL_ART_PLACEMENT |

**Format ของ PoseStamped:**
```
position.x = pixel error จากกึ่งกลางภาพ  (+ = เป้าอยู่ทางขวา)
position.y = pixel error จากกึ่งกลางภาพ  (+ = เป้าอยู่ข้างล่าง)
position.z = depth ที่จุดกึ่งกลาง bbox    (หน่วย mm)
frame_id   = 'base_link'
```

> ⚠️ **ไม่ใช่เมตร** — state machine ใช้ pixel error โดยตรง (tolerance=20px, gain=0.001)

### Topics ที่ subscribe

| Topic | Type | หน้าที่ |
|-------|------|---------|
| `/system_status` | String JSON | รับ macro_state → switch model อัตโนมัติ |
| `/weapon_selection` | String JSON | fallback class switch (spearhead→fist) |

---

## 3. ทดสอบ YOLO node โดยไม่มี robot

**Terminal 1 — รัน node:**
```bash
python3 yolo_node.py
```

**Terminal 2 — จำลอง state machine:**
```bash
# เปิด WEAPON_CLUB (Task 1)
ros2 topic pub /system_status std_msgs/msg/String \
  '{data: "{\"macro_state\": \"WEAPON_CLUB\", \"imu_pitch\": 0.0, \"hardware_error\": \"None\"}"}'

# เปิด MEIHUA (Task 2)
ros2 topic pub /system_status std_msgs/msg/String \
  '{data: "{\"macro_state\": \"MEIHUA_FOREST_EXECUTION\", \"imu_pitch\": 0.0, \"hardware_error\": \"None\"}"}'

# จำลอง fallback spearhead → fist
ros2 topic pub /weapon_selection std_msgs/msg/String \
  '{data: "{\"class\": \"fist\", \"index\": 1}"}'
```

**Terminal 3 — ดู output:**
```bash
ros2 topic echo /vision/task1_target
```

ถ้า detect ได้จะเห็น:
```
pose:
  position:
    x: -45.2    ← เป้าอยู่ซ้าย 45px
    y: 12.1
    z: 620.0    ← ห่าง 620mm
```

---

## 4. Vision Bridge Node (Flask UI ↔ ROS2)

เชื่อมระหว่าง Flask UI กับ state machine ของ robot

```bash
source /opt/ros/humble/setup.bash
python3 vision_bridge_node.py
```

### Topics ที่ bridge จัดการ

**publish → robot:**
| Topic | Type | เมื่อไร |
|-------|------|---------|
| `/mission_plan` | String JSON | กด START ใน UI |
| `/emergency_stop` | Bool | กด DISENGAGE (True) / RESET (False) |

**subscribe ← robot:**
| Topic | Type | หน้าที่ |
|-------|------|---------|
| `/system_status` | String JSON | แสดง state ใน UI |

**Format `/mission_plan`:**
```json
{
  "command": "start_auto",
  "team_color": "BLUE",
  "yolo_conf": 0.75,
  "weapon_club_setup": {"head_index": 1},
  "meihua_forest_path": [2, 5, 8, 11],
  "martial_art_zone": {"place_target": "middle"}
}
```

---

## 5. รันระบบทั้งหมด (วันแข่ง)

```bash
# Terminal 1 — Flask UI
cd ~/vision/R2-ABU && ./start_abu_vision.sh

# Terminal 2 — YOLO node (เจ้าของกล้อง)
source /opt/ros/humble/setup.bash
python3 ~/r2_vision/yolo_node.py

# Terminal 3 — Vision Bridge
source /opt/ros/humble/setup.bash
python3 ~/r2_vision/vision_bridge_node.py

# ปิดทุกอย่าง
cd ~/vision/R2-ABU && ./stop_all.sh
```

**Flask UI เปิดที่:** `http://localhost:5000` หรือ `http://<IP>:5000`

---

## 6. Class names ในแต่ละ model

| Model | Classes |
|-------|---------|
| `weapon.onnx` | `weapondetect`, `fist`, `hand`, **`spearhead`** |
| `cube.pt` | `blue_cube`, `red_cube` |
| `martial.pt` | — (ยังไม่มี) |

---

## 7. ROS_DOMAIN_ID

ต้องตรงกันทั้ง vision computer และ robot computer:
```bash
export ROS_DOMAIN_ID=0   # ใส่ใน ~/.bashrc ด้วย
```

---

## 8. แก้ปัญหา

| ปัญหา | สาเหตุ | วิธีแก้ |
|--------|--------|---------|
| กล้องไม่ขึ้น | process อื่นถือกล้อง | `fuser /dev/video*` แล้ว kill |
| errno=5 | hardware USB error | ถอดเสียบ USB ใหม่ |
| errno=16 | device busy | kill process ที่ถือกล้องอยู่ |
| จอเขียว | USB 2.0 (480M) | ใช้พอร์ต USB 3.0 (5000M) |
| OpenCV ค้าง | imshow ไม่ได้รันบน main thread | ใช้ `yolo_node.py` เวอร์ชันใหม่ |
| YOLO กิน CPU 100% | inference loop ไม่มี cap | ปัจจุบัน cap ที่ 5fps แล้ว |

---

## 9. Checklist ก่อนแข่ง

- [ ] กล้อง USB 3.0 (5000M) — ตรวจด้วย `lsusb -t | grep video`
- [ ] `python3 camera_test.py` — เห็นภาพ FPS > 20
- [ ] `python3 yolo_node.py` — OpenCV window ขึ้น ไม่ค้าง
- [ ] จำลอง WEAPON_CLUB → เอาหอกมาหน้ากล้อง → เห็น bbox + ค่า x/z
- [ ] จำลอง fallback `/weapon_selection {"class":"fist"}` → node switch detect fist
- [ ] `ROS_DOMAIN_ID` ตรงกับ robot
- [ ] Flask UI เปิดได้ → กด START → `/mission_plan` ถูก publish
