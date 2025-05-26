import cv2
import numpy as np
from typing import Dict, Any
from depth_estimator import DepthEstimator
from object_segmenter import ObjectSegmenter
from occupancy_grid import occupancy_grid

class DataProcessor:
    def __init__(self):
        self.state: Dict[str, Any] = {
            "latest_frame": None,
            "latest_depth": None,
            "latest_depth_vis": None,
            "latest_detections": None,
            "latest_detection_vis": None,
            "latest_map_vis": None,
            "is_ground_truth": False,
            "groundtruth": None,
        }
        self.depth_estimator = DepthEstimator(model_type="MiDaS_small")
        self.object_segmenter = ObjectSegmenter(model_path="yolov8s-seg.pt")

    def process_rgb_frame(self, frame: np.ndarray) -> None:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid frame shape: {frame.shape if frame is not None else None}")

        self.state["latest_frame"] = frame

        # 1. Depth estimation
        depth_map, depth_error = self.depth_estimator.estimate_depth(frame)
        if depth_error or depth_map is None:
            self.state["latest_depth"] = None
            self.state["latest_depth_vis"] = None
        else:
            self.state["latest_depth"] = depth_map
            self.state["latest_depth_vis"] = self.depth_estimator.get_visualization(depth_map)

        # 2. Object detection / segmentation
        detections, det_error = self.object_segmenter.detect_and_segment(frame)
        if det_error or detections is None:
            self.state["latest_detections"] = None
            self.state["latest_detection_vis"] = None
        else:
            self.state["latest_detections"] = detections
            det_vis, _ = self.object_segmenter.get_visualization(frame, detections)
            self.state["latest_detection_vis"] = det_vis

        # 3. Update occupancy grid (if pose & depth available)
        if (
            depth_map is not None and
            self.state["groundtruth"] is not None and
            "position" in self.state["groundtruth"] and
            "rotation" in self.state["groundtruth"]
        ):
            pos = self.state["groundtruth"]["position"]
            rot = self.state["groundtruth"]["rotation"]
            
            # Store current position and rotation for FOV visualization
            occupancy_grid.current_pos = pos
            occupancy_grid.current_rot = rot
            
            occupancy_grid.update_from_depth(depth_map, pos, rot)
            self.state["latest_map_vis"] = occupancy_grid.get_visualization()
            self.state["is_ground_truth"] = False

    def process_ground_truth(self, depth_map: np.ndarray) -> None:
        if depth_map is None:
            return

        self.state["latest_depth"] = depth_map
        self.state["latest_depth_vis"] = self.depth_estimator.get_visualization(depth_map)

        if self.state["groundtruth"] and "position" in self.state["groundtruth"]:
            pos = self.state["groundtruth"]["position"]
            rot = self.state["groundtruth"]["rotation"]
            
            # Store current position and rotation for FOV visualization
            occupancy_grid.current_pos = pos
            occupancy_grid.current_rot = rot
            
            occupancy_grid.update_from_depth(depth_map, pos, rot)
            self.state["latest_map_vis"] = occupancy_grid.get_visualization()
            self.state["is_ground_truth"] = True

    def process_groundtruth_data(self, data: Dict[str, Any]) -> None:
        try:
            self.state["groundtruth"] = {
                "timestamp": round(data["timestamp"], 2),
                "position": [round(v, 2) for v in data["position"]],
                "rotation": [round(v, 2) for v in data["rotation"]],
                "velocity": [round(v, 2) for v in data["velocity"]],
                "angular_velocity": [round(v, 2) for v in data["angular_velocity"]],
            }
        except Exception as e:
            print("Error processing groundtruth data:", e)

    def get_state(self) -> Dict[str, Any]:
        return self.state

# Singleton
data_processor = DataProcessor()
__all__ = ['data_processor']
