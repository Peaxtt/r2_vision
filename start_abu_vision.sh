#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  ABU Robocon 2026 — Vision & Mission Control UI
#  เปิด Flask web server สำหรับ operator ใช้ควบคุม robot
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$SCRIPT_DIR/ui"
APP_FILE="$APP_DIR/app.py"
PID_FILE=/tmp/abu_flask.pid
LOG_FILE=/tmp/abu_flask.log
IP=$(hostname -I | awk '{print $1}')

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║     ABU Robocon 2026 — Vision UI         ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""
echo "  App    : $APP_FILE"
echo "  Log    : $LOG_FILE"
echo ""

if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
    echo "  [!] Already running — PID $(cat $PID_FILE)"
    echo "  [i] Open  : http://$IP:5000"
    echo "  [i] Stop  : ./stop.sh"
    echo ""
    exit 0
fi

cd "$APP_DIR"
nohup python3 -u app.py > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

sleep 1
if kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
    echo "  [✓] Server started — PID $(cat $PID_FILE)"
    echo ""
    echo "  ┌─ เปิดในเครื่องนี้ ──────────────────────────┐"
    echo "  │  http://localhost:5000                       │"
    echo "  ├─ เปิดใน iPad / เครื่องอื่นใน network ───────┤"
    echo "  │  http://$IP:5000                  │"
    echo "  └──────────────────────────────────────────────┘"
    echo ""
    echo "  ดู Flask log : tail -f $LOG_FILE"
    echo "  ดู App log   : tail -f /tmp/abu_vision.log"
    echo "  ปิด server   : ./stop.sh"
else
    echo "  [✗] Failed to start — ดู log:"
    echo "      cat $LOG_FILE"
fi
echo ""
