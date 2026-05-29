#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  ABU Robocon 2026 — ปิดทุก process ที่เกี่ยวกับระบบ
#  (Flask UI + camera pipeline + YOLO)
# ─────────────────────────────────────────────────────────────

PID_FILE=/tmp/abu_flask.pid

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   ABU Robocon 2026 — Stop All            ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

KILLED=0

# 1. kill via PID file ถ้ามี
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "  [✓] Flask server (PID $PID)"
        KILLED=$((KILLED+1))
    fi
    rm -f "$PID_FILE"
fi

# 2. kill process ที่รัน app.py อยู่ (เผื่อเปิดด้วยมือ)
for PID in $(pgrep -f "app\.py"); do
    kill "$PID" 2>/dev/null && echo "  [✓] Flask process (PID $PID)" && KILLED=$((KILLED+1))
done

# 3. kill อะไรก็ตามที่ยึด port 5000
PORT_PID=$(lsof -ti :5000 2>/dev/null)
if [ -n "$PORT_PID" ]; then
    kill $PORT_PID 2>/dev/null && echo "  [✓] Port 5000 process (PID $PORT_PID)" && KILLED=$((KILLED+1))
fi

# 4. kill ROS2 nodes
for PATTERN in "vision_bridge_node" "yolo_detection_node" "flask_cam_sub"; do
    for PID in $(pgrep -f "$PATTERN"); do
        kill -9 "$PID" 2>/dev/null && echo "  [✓] $PATTERN (PID $PID)" && KILLED=$((KILLED+1))
    done
done

# 5. kill ros2 daemon ถ้ามี
for PID in $(pgrep -f "ros2-daemon"); do
    kill "$PID" 2>/dev/null && echo "  [✓] ros2 daemon (PID $PID)" && KILLED=$((KILLED+1))
done

# 6. ล้าง PID file และ temp log
rm -f /tmp/abu_flask.pid /tmp/yolo_test.log /tmp/yolo_node.log 2>/dev/null

echo ""
if [ "$KILLED" -eq 0 ]; then
    echo "  [i] ไม่มี process ที่ต้องปิด"
else
    echo "  [✓] ปิดทั้งหมด $KILLED process"
fi
echo ""
