import threading
import cv2
import numpy as np
import traceback
from server import app, state
from config import SHOW_WINDOWS

def draw_vector_row(y, label, values, canvas, color=(200, 255, 200)):
    text = f"{label:<16} x={values[0]:>6}  y={values[1]:>6}  z={values[2]:>6}"
    cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_imu_display(state):
    h, w = 300, 500
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y = 20

    # Title
    cv2.putText(canvas, "📡 VEHICLE STATE MONITOR", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y += 20

    # Groundtruth
    gt = state.get("groundtruth")
    if gt:
        cv2.putText(canvas, f"Groundtruth @ {gt['timestamp']}s", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
        y += 20
        draw_vector_row(y, "Position", gt["position"], canvas); y += 20
        draw_vector_row(y, "Rotation", gt["rotation"], canvas); y += 20
        draw_vector_row(y, "Velocity", gt["velocity"], canvas); y += 20
        draw_vector_row(y, "Angular Vel", gt["angular_velocity"], canvas); y += 30

    # IMU
    imu = state.get("imu")
    if imu:
        cv2.putText(canvas, f"IMU @ {imu['timestamp']}s", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        y += 20
        draw_vector_row(y, "Acceleration", imu["acceleration"], canvas); y += 20
        draw_vector_row(y, "Angular Vel", imu["angular_velocity"], canvas); y += 20
        draw_vector_row(y, "Orientation", imu["orientation"], canvas); y += 20

    return canvas

def display_loop():
    while True:
        try:
            if state["latest_frame"] is not None:
                cv2.imshow("Unity Camera Feed", state["latest_frame"])

            if state["latest_bev_frame"] is not None:
                cv2.imshow("Top-Down BEV View", state["latest_bev_frame"])

            if state["latest_depth"] is not None and state["latest_depth"].ndim == 2:
                depth_vis = (state["latest_depth"] * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
                cv2.imshow("Depth Estimation", depth_colored)

            if state.get("groundtruth") or state.get("imu"):
                canvas = draw_imu_display(state)
                cv2.imshow("📡 IMU Monitor", canvas)

            if state.get("occupancy_grid") is not None:
                cv2.imshow("🗺️ Occupancy Grid", state["occupancy_grid"])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print("Display error:", e)
            traceback.print_exc()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("🟢 Flask server running at http://127.0.0.1:5000")
    print("🎥 RGB stream:         http://127.0.0.1:5000/video/rgb")
    print("🌊 Depth stream:       http://127.0.0.1:5000/video/depth")
    print("⬛ BEV stream:         http://127.0.0.1:5000/video/bev")
    print("📡 IMU endpoint:       http://127.0.0.1:5000/imu")
    print("🛰️  Groundtruth:       http://127.0.0.1:5000/groundtruth")
    print("📊 Vehicle Data:       http://127.0.0.1:5000/data/vehicle")

    if SHOW_WINDOWS:
        threading.Thread(target=display_loop, daemon=True).start()

    app.run(host="0.0.0.0", port=5000)
