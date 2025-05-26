import numpy as np
import cv2
from typing import Tuple, List

class SimpleMapper:
    def __init__(self,
                 grid_size: float = 0.2,
                 map_dim: Tuple[int, int] = (100, 100),
                 camera_height: float = 1.6):
        self.grid_size = grid_size  # meters per cell
        self.map_w, self.map_h = map_dim
        self.camera_height = camera_height

        # Camera intrinsics (for 256x256 image)
        image_width = 256
        image_height = 256
        sensor_width = 36  # mm
        sensor_height = 24  # mm
        focal_length = 50  # mm

        self.fx = (image_width * focal_length) / sensor_width
        self.fy = (image_height * focal_length) / sensor_height
        self.cx = image_width / 2
        self.cy = image_height / 2

        # Initialize occupancy grid
        self.occupancy_grid = np.zeros((self.map_h, self.map_w), dtype=np.uint8)

    def project_detections(self, detections: List, frame_shape: Tuple[int, int]):
        """Project detection bounding boxes onto the ground plane"""
        # Clear previous grid
        self.occupancy_grid.fill(0)
        
        h, w = frame_shape[:2]
        center_x = self.map_w // 2
        center_y = self.map_h // 2

        for det in detections:
            # Get bottom center point of bounding box
            x1, y1, x2, y2 = det.bbox
            bottom_center_x = (x1 + x2) / 2
            bottom_center_y = y2  # Use bottom of bounding box

            # Project to ground plane
            # Using similar triangles principle
            z = self.camera_height * self.fy / (bottom_center_y - self.cy)
            x = (bottom_center_x - self.cx) * z / self.fx

            # Convert to grid coordinates
            grid_x = int(x / self.grid_size) + center_x
            grid_y = int(z / self.grid_size) + center_y

            # Check if point is within grid bounds
            if 0 <= grid_x < self.map_w and 0 <= grid_y < self.map_h:
                # Draw a small circle at the projected point
                cv2.circle(self.occupancy_grid, (grid_x, grid_y), 2, 255, -1)

    def get_visualization(self) -> np.ndarray:
        """Get visualization of the occupancy grid"""
        # Convert to 3-channel image
        vis = np.stack([self.occupancy_grid]*3, axis=-1)
        
        # Resize for better visualization
        vis = cv2.resize(vis, (300, 300), interpolation=cv2.INTER_NEAREST)
        
        # Apply colormap
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        
        # Add grid lines
        grid_spacing = 10
        for i in range(0, vis.shape[0], grid_spacing):
            cv2.line(vis, (0, i), (vis.shape[1], i), (50, 50, 50), 1)
        for i in range(0, vis.shape[1], grid_spacing):
            cv2.line(vis, (i, 0), (i, vis.shape[0]), (50, 50, 50), 1)
            
        return vis

# Singleton instance
simple_mapper = SimpleMapper()
__all__ = ['simple_mapper'] 