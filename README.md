# R2 Vision — ABU Robocon 2026

Vision system สำหรับหุ่นยนต์ R2 ประกอบด้วย YOLO detection node + ROS2 bridge + Flask UI

---

## โครงสร้างไฟล์

```
r2_vision/
├── camera_test.py          ← ทดสอบกล้องโดดๆ (ไม่ต้องมี ROS)
├── yolo_node.py            ← OpenVINO detection + ROS2 publisher (ไฟล์หลัก)
├── model_loader.py         ← OpenVINO-first model loader (auto-convert .pt/.onnx → IR)
├── processors.py           ← per-task processing (1 model = 1 pipeline)
├── vision_bridge_node.py   ← เชื่อม Flask UI ↔ ROS2 state machine
├── start_abu_vision.sh     ← สั่งเปิด Flask UI
├── stop_all.sh             ← ปิดทุก process
├── requirements.txt        ← Python dependencies (+ openvino)
├── models/
│   ├── yolonew_openvino_model/ ← Task 1: weapon model (OpenVINO IR — committed)
│   ├── cube_openvino_model/    ← Task 2+3: cube model (OpenVINO IR — committed)
│   └── centroid_config.txt ← weapon centroid offset + filter thresholds (tuned)
├── weapon_detection/       ← standalone weapon reference repo (gitignored)
│   └── weapon_detection_vino.py     ← original standalone script
├── Cube-Detection/         ← standalone cube reference repo (gitignored)
│   ├── grid_detection.py            ← original 3×3 grid occupancy script
│   └── best.pt                      ← cube weights (source of cube IR)
└── ui/                     ← Flask Mission Control UI
    ├── app.py              ← Flask entry point
    ├── api/                ← REST API endpoints
    │   ├── control.py      ← mission commands
    │   ├── telemetry.py    ← camera stream
    │   ├── diagnostics.py  ← system test
    │   └── mission.py      ← path planning API
    ├── core/               ← business logic
    │   ├── state.py        ← robot_state shared dict
    │   ├── pathfinding.py  ← Meihua Forest path planner
    │   └── logger.py       ← rotating log setup
    ├── templates/
    │   └── index.html      ← operator UI (TailwindCSS)
    └── static/             ← icons, manifest (PWA)
```

---

## ความต้องการ

```bash
pip install -r requirements.txt   # pyrealsense2 ultralytics opencv-python numpy openvino onnx onnxruntime pyyaml
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

## 2. YOLO Detection Node (OpenVINO)

ไฟล์หลัก — เปิดกล้อง, รัน OpenVINO YOLO, แสดง OpenCV window, publish ไปยัง ROS2

**1 model = 1 pipeline.** แต่ละ macro_state มี processor ของตัวเอง (ดู `processors.py`):

| State | Processor | Model | Output |
|-------|-----------|-------|--------|
| `WEAPON_CLUB` | `WeaponProcessor` | weapon (OpenVINO IR) | **String JSON** — mm coords + placement index |
| `MEIHUA_FOREST_EXECUTION` | `SimpleBestProcessor` | cube (OpenVINO IR) | `PoseStamped` — pixel error + depth (เป้าใน forest) |
| `MARTIAL_ART_PLACEMENT` | `GridProcessor` | cube (OpenVINO IR) | **String JSON** — 3×3 grid EMPTY/FULL (arena) |

> ทั้ง forest (task2) และ grid (task3) ใช้ **cube model เดียวกัน** (`blue_cube`/`red_cube`).
> forest = หา cube ที่ดีที่สุด → pixel error; grid = หาตาราง 3×3 แล้วเช็คว่าแต่ละช่องว่าง/มี cube.
>
> ทุก model โหลดเป็น **OpenVINO IR**. ถ้าใส่ `.pt`/`.onnx` จะ auto-convert เป็น IR
> ครั้งแรก (cache ไว้ข้างไฟล์ต้นทาง) แล้วโหลด IR ตรงๆ ครั้งถัดไป.
> Task ไหนไม่มี model → ตกเป็น `AbstractProcessor` อัตโนมัติ.

### รัน
```bash
source /opt/ros/humble/setup.bash
python3 yolo_node.py
```

### ปรับค่าผ่าน ROS2 arguments
```bash
python3 yolo_node.py --ros-args \
  -p model_weapon:=/path/to/yolonew_openvino_model \
  -p device:=CPU \        # OpenVINO device: CPU / GPU / AUTO
  -p rotate:=270 \        # หมุนภาพตามการติดตั้งกล้อง (0/90/180/270)
  -p imgsz:=640 \
  -p conf:=0.5 \
  -p grid_use_depth:=true # task3 grid: ใช้ depth ยืนยันช่อง FULL ด้วย (default true)
```

### Model paths
แก้ใน `yolo_node.py` บรรทัดนี้:
```python
MODEL_WEAPON  = os.path.join(MODELS_DIR, 'yolonew_openvino_model')
MODEL_MEIHUA  = os.path.join(MODELS_DIR, 'cube_openvino_model')   # forest cube
MODEL_MARTIAL = os.path.join(MODELS_DIR, 'cube_openvino_model')   # grid (same model)
```

### Topics ที่ publish

| Topic | Type | เมื่อไร |
|-------|------|---------|
| `/vision/task1_target` | **String JSON** | WEAPON_CLUB — รายการ detection ทั้งหมด (mm + index) |
| `/vision/task2_target` | PoseStamped | MEIHUA_FOREST_EXECUTION + detect cube ใน forest |
| `/vision/task3_target` | **String JSON** | MARTIAL_ART_PLACEMENT — 3×3 grid occupancy |

**Format `/vision/task1_target` (WeaponProcessor):**
```json
{
  "task": "WEAPON_CLUB",
  "stamp": 1234567890,
  "detections": [
    {"index": 0, "class": "spearhead", "conf": 0.91,
     "x_from_principle_mm": -134.2, "y_from_principle_mm": 18.5,
     "z_depth_mm": 920, "abs_dist_mm": 931.0}
  ]
}
```
- `index` = placement slot 0–5 (`-1` ถ้าไม่อยู่ใน layout)
- พิกัด mm วัดจาก principal point: X+ ขวา, Y+ ล่าง, Z+ ลึกเข้าฉาก
- ค่า mm เป็น `null` ถ้าไม่มี depth ที่จุด centroid

**Format `/vision/task2_target` (SimpleBestProcessor — PoseStamped):**
```
position.x = pixel error จากกึ่งกลางภาพ  (+ = เป้าอยู่ทางขวา)
position.y = pixel error จากกึ่งกลางภาพ  (+ = เป้าอยู่ข้างล่าง)
position.z = depth ที่จุดกึ่งกลาง bbox    (หน่วย mm)
frame_id   = 'base_link'
```

**Format `/vision/task3_target` (GridProcessor — 3×3 grid occupancy):**
```json
{
  "task": "MARTIAL_ART_PLACEMENT",
  "stamp": 1234567890,
  "grid_found": true,
  "corners": [[156,195],[363,76],[483,284],[275,403]],
  "bbox": [gx, gy, gw, gh],
  "grid": [["EMPTY","FULL","EMPTY"],
           ["EMPTY","EMPTY","EMPTY"],
           ["EMPTY","EMPTY","FULL"]],
  "cells": [
    {"row":0,"col":0,"index":0,"status":"EMPTY",
     "center_px":[220,140],"z_mm":1000,"cube":false,"depth":false}
  ]
}
```
- `corners` = มุมตาราง 4 จุด เรียง TL,TR,BR,BL (รองรับตารางเอียง/หมุน)
- `bbox` = กรอบสี่เหลี่ยมแนวแกนครอบ corners (ไว้ compat)
- `grid` = ตาราง 3×3 row-major, `"EMPTY"` (ช่องว่าง) หรือ `"FULL"` (มี cube)
- `cells[].index` = 0–8 (row*3+col), `center_px` = จุดกึ่งกลางช่อง (back-project จาก homography),
  `z_mm` = depth ที่ช่องนั้น, `cube`/`depth` = สัญญาณไหนตัดสินว่า FULL
- ถ้าหาตารางไม่เจอ → `grid_found=false`, `grid`/`cells` ว่าง

> **Robust กับตารางหมุน:** หา grid เป็น quadrilateral (contour + minAreaRect fallback)
> แล้ว warp ด้วย homography เป็นตาราง canonical 3×3 — ไม่ยึดแกนนอน/ตั้ง.
> ช่อง FULL ตัดสินจาก 2 สัญญาณ (OR): cube ที่ project เข้าช่อง **หรือ** depth ที่ช่อง
> นั้นมีค่าใกล้ระนาบกระดาน (รูว่างจะเห็น background ไกล/เกิน max_depth = invalid).
> ปิด depth ได้ด้วย `-p grid_use_depth:=false`.
> ⚠️ ตารางสี่เหลี่ยมไม่มีจุดอ้างอิงทิศ → (row,col) กำหนดได้แค่ตามความสมมาตร 4 ทิศ
> ของตาราง; **ความว่าง/เต็มถูกเสมอ** แต่ป้าย row/col อาจหมุนตามตาราง.

> ⚠️ Task 1 (mm + index JSON) และ Task 3 (grid JSON) ไม่ใช่ PoseStamped —
> state machine ต้องอ่าน JSON. Task 2 ยังเป็น pixel error เหมือนเดิม
> (tolerance=20px, gain=0.001).

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
ros2 topic echo /vision/task1_target   # std_msgs/String (JSON)
```

ถ้า detect ได้จะเห็น (Task 1 = weapon, mm + placement index):
```
data: '{"task": "WEAPON_CLUB", "stamp": 1234567890, "detections":
  [{"index": 0, "class": "spearhead", "conf": 0.91,
    "x_from_principle_mm": -134.2, "y_from_principle_mm": 18.5,
    "z_depth_mm": 920, "abs_dist_mm": 931.0}]}'
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

| Model | Format | Classes | ใช้ใน |
|-------|--------|---------|-------|
| `yolonew_openvino_model/` | OpenVINO IR | `weapondetect`, `fist`, `hand`, **`spearhead`** | Task 1 (weapon) |
| `cube_openvino_model/` | OpenVINO IR | `blue_cube`, `red_cube` | Task 2 (forest) + Task 3 (grid) |

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
