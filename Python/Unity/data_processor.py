import cv2
import numpy as np
from mapping import OccupancyGridMapper

# Initialize once (global)
grid_mapper = OccupancyGridMapper()

# Define label → int mapping
labels_map = {
    "car": 1,
    "person": 2,
    # Add more classes here if needed
}

def process_rgb_frame(frame, depth_estimator, streamer, state):
    state["latest_frame"] = frame
    state["latest_depth"] = depth_estimator.estimate_depth(frame)

    # Stream original RGB
    streamer.update_frame("rgb", frame)

    # Stream colorized depth
    if state["latest_depth"] is not None:
        depth_vis = (state["latest_depth"] * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        streamer.update_frame("depth", depth_colored)

        # ➕ Occupancy Mapping
        detections = state.get("latest_detection", [])  # List of {bbox, label}
        grid_mapper.clear()
        grid_mapper.update_from_depth(state["latest_depth"])
        grid_mapper.add_detections(detections, state["latest_depth"], labels_map)
        state["occupancy_grid"] = grid_mapper.get_visualization()
        streamer.update_frame("occupancy", state["occupancy_grid"])


def process_bev_frame(frame, streamer, state):
    state["latest_bev_frame"] = frame
    streamer.update_frame("bev", frame)


def process_groundtruth_data(data, state):
    try:
        state["groundtruth"] = {
            "timestamp": round(data["timestamp"], 2),
            "position": [round(v, 2) for v in data["position"]],
            "rotation": [round(v, 2) for v in data["rotation"]],
            "velocity": [round(v, 2) for v in data["velocity"]],
            "angular_velocity": [round(v, 2) for v in data["angular_velocity"]],
        }
    except Exception as e:
        print("Error processing groundtruth data:", e)


def process_imu_data(data, state):
    try:
        state["imu"] = {
            "timestamp": round(data["timestamp"], 2),
            "acceleration": [round(a, 2) for a in data["acceleration"]],
            "angular_velocity": [round(g, 2) for g in data["angularVelocity"]],
            "orientation": [round(o, 2) for o in data["orientation"]],
        }
    except Exception as e:
        print("Error processing IMU data:", e)
