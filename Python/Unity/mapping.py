import numpy as np
import cv2

class OccupancyGridMapper:
    def __init__(self, grid_size=(100, 100), scale=0.1):
        """
        Args:
            grid_size: (width, height) in cells
            scale: meters per cell (e.g., 0.1 → each cell = 10cm x 10cm)
        """
        self.grid_size = grid_size
        self.scale = scale
        self.grid = np.zeros(grid_size, dtype=np.uint8)  # 0=free, 100=occupied
        self.semantic_map = np.zeros(grid_size, dtype=np.uint8)  # optional

        # Camera intrinsics (you can replace with actual calibration)
        fx, fy = 320, 320
        cx, cy = 320, 240
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]])

    def clear(self):
        self.grid.fill(0)
        self.semantic_map.fill(0)

    def update_from_depth(self, depth_map, camera_pose=None):
        """
        Projects depth image into 3D points and builds top-down occupancy.
        depth_map: HxW depth in meters
        camera_pose: optional (x, y, z, theta) for global alignment (not used here)
        """
        H, W = depth_map.shape
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        zs = depth_map
        xs = (xs - self.K[0, 2]) * zs / self.K[0, 0]
        ys = (ys - self.K[1, 2]) * zs / self.K[1, 1]

        points = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)

        # Project to top-down 2D grid
        for pt in points:
            x_m, y_m = pt[0], pt[2]  # horizontal and forward
            gx = int(self.grid_size[0] // 2 + x_m / self.scale)
            gy = int(y_m / self.scale)
            if 0 <= gx < self.grid_size[0] and 0 <= gy < self.grid_size[1]:
                self.grid[gy, gx] = 100

    def add_detections(self, detections, depth_map, labels_map):
        """
        Projects object detections into top-down semantic map.
        detections: list of dicts with bbox=[x1,y1,x2,y2], label='car', etc.
        labels_map: dict of class → int (e.g., {'car': 1, 'person': 2})
        """
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = det['label']
            mask = np.zeros_like(depth_map, dtype=bool)
            mask[y1:y2, x1:x2] = True

            if np.count_nonzero(mask) == 0:
                continue

            mean_depth = np.mean(depth_map[mask])
            if np.isnan(mean_depth):
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            x = (cx - self.K[0, 2]) * mean_depth / self.K[0, 0]
            y = (cy - self.K[1, 2]) * mean_depth / self.K[1, 1]
            z = mean_depth

            gx = int(self.grid_size[0] // 2 + x / self.scale)
            gy = int(z / self.scale)

            if 0 <= gx < self.grid_size[0] and 0 <= gy < self.grid_size[1]:
                self.semantic_map[gy, gx] = labels_map.get(label, 255)

    def get_visualization(self):
        vis = np.zeros((*self.grid.shape, 3), dtype=np.uint8)
        vis[self.grid > 0] = (100, 100, 100)
        unique_labels = np.unique(self.semantic_map)
        for label in unique_labels:
            if label == 0:
                continue
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            vis[self.semantic_map == label] = color
        return vis
