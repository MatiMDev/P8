import numpy as np
import cv2
from typing import Tuple

class BEVMapper:
    def __init__(self,
                 grid_size: float = 0.2,
                 map_dim: Tuple[int, int] = (100, 100),
                 camera_height: float = 1.6):
        self.grid_size = grid_size  # meters per cell
        self.map_w, self.map_h = map_dim
        self.camera_height = camera_height

        # Camera intrinsics (updated for 256x256, 50mm focal, 36x24mm sensor)
        image_width = 256
        image_height = 256
        sensor_width = 36  # mm
        sensor_height = 24  # mm
        focal_length = 50  # mm

        self.fx = (image_width * focal_length) / sensor_width  # ≈ 355.56
        self.fy = (image_height * focal_length) / sensor_height  # ≈ 533.33
        self.cx = image_width / 2  # 128
        self.cy = image_height / 2  # 128

        self.occupancy_grid = np.zeros((self.map_h, self.map_w), dtype=np.uint8)

    def update_from_depth(self, depth_map: np.ndarray, position, rotation_deg):
        # Optional decay of previous map data
        self.occupancy_grid = np.clip(self.occupancy_grid - 1, 0, 255)

        point_cloud = self._depth_to_point_cloud(depth_map)

        # Filter for near-ground points
        ground_mask = np.abs(point_cloud[:, 1] + self.camera_height) < 0.3
        ground_points = point_cloud[ground_mask]

        # Transform to world coordinates
        world_points = self._transform_points(ground_points, position, rotation_deg)

        # Update persistent BEV grid
        self._update_grid(world_points)

    def _depth_to_point_cloud(self, depth_map: np.ndarray) -> np.ndarray:
        h, w = depth_map.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_map.flatten()
        x = (i.flatten() - self.cx) * z / self.fx
        y = (j.flatten() - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=-1)  # shape: (N, 3)

    def _transform_points(self, points: np.ndarray, position, rotation_deg) -> np.ndarray:
        # Convert Euler angles from degrees to radians
        rx, ry, rz = np.radians(rotation_deg)

        # Rotation matrices
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz),  np.cos(rz), 0],
            [0,           0,          1]
        ])
        Ry = np.array([
            [ np.cos(ry), 0, np.sin(ry)],
            [0,           1, 0         ],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rx = np.array([
            [1, 0,          0         ],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx),  np.cos(rx)]
        ])
        R = Rz @ Ry @ Rx
        rotated = points @ R.T
        translated = rotated + np.array(position)
        return translated

    def _update_grid(self, world_points: np.ndarray):
        center_x = self.map_w // 2
        center_y = self.map_h // 2

        x_coords = world_points[:, 0]
        z_coords = world_points[:, 2]

        u = np.floor(x_coords / self.grid_size).astype(int) + center_x
        v = np.floor(z_coords / self.grid_size).astype(int) + center_y

        valid = (u >= 0) & (u < self.map_w) & (v >= 0) & (v < self.map_h)
        u, v = u[valid], v[valid]

        # Accumulate occupancy evidence
        self.occupancy_grid[v, u] = np.clip(self.occupancy_grid[v, u] + 20, 0, 255)

    def get_visualization(self) -> np.ndarray:
        vis = np.stack([self.occupancy_grid]*3, axis=-1)  # grayscale → 3-channel
        vis = cv2.resize(vis, (300, 300), interpolation=cv2.INTER_NEAREST)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        return vis

# Singleton
mapper = BEVMapper()
__all__ = ['mapper']
