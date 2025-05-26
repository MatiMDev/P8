import threading
import cv2
import numpy as np
import traceback
from server import app
from data_processor import data_processor
from config import SHOW_WINDOWS
from trajectory_visualizer_pygame import trajectory_visualizer_pygame
from mapper import mapper  # Import the BEV mapper

def draw_vector_row(y, label, values, canvas, color=(200, 255, 200)):
    text = f"{label:<16} x={values[0]:>6}  y={values[1]:>6}  z={values[2]:>6}"
    cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_groundtruth_display(state):
    h, w = 300, 500
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y = 20

    # Title
    cv2.putText(canvas, "VEHICLE STATE MONITOR", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y += 20

    # Groundtruth
    gt = state.get("groundtruth")
    if gt:
        cv2.putText(canvas, f"Groundtruth @ {gt['timestamp']}s", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
        y += 20
        draw_vector_row(y, "Position", gt["position"], canvas); y += 20
        draw_vector_row(y, "Rotation", gt["rotation"], canvas); y += 20
        draw_vector_row(y, "Velocity", gt["velocity"], canvas); y += 20
        draw_vector_row(y, "Angular Vel", gt["angular_velocity"], canvas); y += 20

    return canvas

def display_loop():
    while True:
        try:
            state = data_processor.get_state()
            
            # Display RGB frame
            if state["latest_frame"] is not None:
                cv2.imshow("Unity Camera Feed", state["latest_frame"])

            # Display depth visualization
            if state["latest_depth_vis"] is not None:
                title = "Ground Truth Depth" if state["is_ground_truth"] else "Estimated Depth"
                cv2.imshow(title, state["latest_depth_vis"])
                
                # Update BEV map with depth data if we have position and rotation
                if state.get("groundtruth"):
                    gt = state["groundtruth"]
                    depth_map = state["latest_depth_vis"][:,:,0]  # Use first channel of depth visualization
                    mapper.update_from_depth(depth_map, gt["position"], gt["rotation"])
                    
                    # Display BEV map
                    bev_vis = mapper.get_visualization()
                    cv2.imshow("Bird's Eye View Map", bev_vis)

            # Display object detection
            if state["latest_detection_vis"] is not None:
                cv2.imshow("Object Detection", state["latest_detection_vis"])

            # Display ground truth data
            if state.get("groundtruth"):
                canvas = draw_groundtruth_display(state)
                cv2.imshow("\U0001F4E1 Ground Truth Monitor", canvas)
                # Feed trajectory data to PyGame visualizer
                pos = state["groundtruth"].get("position")
                timestamp = state["groundtruth"].get("timestamp")
                if pos and timestamp:
                    trajectory_visualizer_pygame.add_position(pos, timestamp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            traceback.print_exc()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("🟢 Flask server running at http://127.0.0.1:5000")
    print("🎥 RGB stream:         http://127.0.0.1:5000/frame")
    print("🌊 Depth visualization: http://127.0.0.1:5000/frame")
    print("📦 Object detection:    http://127.0.0.1:5000/frame")
    print("🗺️  Occupancy map:      http://127.0.0.1:5000/frame")
    print("📊 Ground truth:        http://127.0.0.1:5000/groundtruth")

    if SHOW_WINDOWS:
        # Start PyGame visualizer in its own thread
        trajectory_visualizer_pygame.start()
        threading.Thread(target=display_loop, daemon=True).start()

    app.run(host="0.0.0.0", port=5000) 