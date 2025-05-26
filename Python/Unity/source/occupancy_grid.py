import numpy as np
from typing import Tuple, List, Optional
import cv2
from dataclasses import dataclass

@dataclass
class GridCell:
    """Represents a cell in the occupancy grid."""
    log_odds: float = 0.0  # Log odds of occupancy
    hits: int = 0          # Number of times this cell was hit
    misses: int = 0        # Number of times this cell was missed

class OccupancyGrid:
    def __init__(self, 
                 grid_size_meters: float = 400.0,
                 resolution: float = 1.0,  # meters per cell (lower resolution for speed)
                 max_range: float = 50.0,  # maximum sensor range in meters
                 log_odds_hit: float = 0.7,
                 log_odds_miss: float = -0.4,
                 log_odds_threshold: float = 0.0):
        """
        Initialize the occupancy grid.
        
        Args:
            grid_size_meters: Size of the grid in meters (assumed square)
            resolution: Size of each cell in meters
            max_range: Maximum sensor range in meters
            log_odds_hit: Log odds value for hits
            log_odds_miss: Log odds value for misses
            log_odds_threshold: Threshold for occupancy
        """
        self.resolution = resolution
        self.max_range = max_range
        self.log_odds_hit = log_odds_hit
        self.log_odds_miss = log_odds_miss
        self.log_odds_threshold = log_odds_threshold
        
        # Calculate grid dimensions
        self.grid_size = int(grid_size_meters / resolution)
        self.grid_center = self.grid_size // 2
        
        # Initialize grid with unknown cells
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=object)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid[i, j] = GridCell()

        # Camera parameters from user (in mm and degrees)
        sensor_width = 36.0
        sensor_height = 24.0
        focal_length = 50.0
        # Calculate FOVs in radians
        self.fov_horizontal = 2 * np.arctan((sensor_width / 2) / focal_length)  # ~0.691 rad (~39.6 deg)
        self.fov_vertical = 2 * np.arctan((sensor_height / 2) / focal_length)   # ~0.471 rad (~26.99 deg)

    def world_to_grid(self, x: float, z: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        grid_x = int(x / self.resolution) + self.grid_center
        grid_z = int(z / self.resolution) + self.grid_center
        return grid_x, grid_z

    def grid_to_world(self, grid_x: int, grid_z: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        x = (grid_x - self.grid_center) * self.resolution
        z = (grid_z - self.grid_center) * self.resolution
        return x, z

    def update_from_depth(self, 
                         depth_map: np.ndarray,
                         position: List[float],
                         rotation: List[float]) -> None:
        """
        Update the occupancy grid using depth data.
        
        Args:
            depth_map: Normalized depth map (0-1)
            position: Current position [x, y, z]
            rotation: Current rotation [pitch, yaw, roll] in radians
        """
        if depth_map is None or depth_map.shape != (256, 256):
            return

        # Subsample for speed
        step = 8  # Only process every 8th pixel
        depth_meters = depth_map * self.max_range
        pos_x, pos_z = self.world_to_grid(position[0], position[2])
        for i in range(0, 256, step):
            for j in range(0, 256, step):
                depth = depth_meters[i, j]
                if depth < 0.1:  # Skip very close points
                    continue

                # Normalize pixel coordinates to [-1, 1]
                x_norm = (j - 128) / 128.0
                z_norm = (i - 128) / 128.0
                
                # Calculate angles in world space
                angle_h = x_norm * (self.fov_horizontal / 2)
                # angle_v = z_norm * (self.fov_vertical / 2)  # Not used for 2D mapping
                
                # Calculate world coordinates of the point (car is always the origin)
                world_x = position[0] + depth * np.sin(angle_h + rotation[1])
                world_z = position[2] + depth * np.cos(angle_h + rotation[1])
                
                # Convert to grid coordinates
                grid_x, grid_z = self.world_to_grid(world_x, world_z)
                
                # Check if point is within grid bounds
                if 0 <= grid_x < self.grid_size and 0 <= grid_z < self.grid_size:
                    # Update the cell
                    cell = self.grid[grid_x, grid_z]
                    cell.log_odds += self.log_odds_hit
                    cell.hits += 1

                # Update cells along the ray
                self._update_ray(pos_x, pos_z, grid_x, grid_z)

    def _update_ray(self, x0: int, z0: int, x1: int, z1: int) -> None:
        """Update cells along a ray using Bresenham's line algorithm."""
        dx = abs(x1 - x0)
        dz = abs(z1 - z0)
        sx = 1 if x0 < x1 else -1
        sz = 1 if z0 < z1 else -1
        err = dx - dz

        while True:
            # Skip the endpoint (already updated)
            if x0 == x1 and z0 == z1:
                break

            # Update cell if within bounds
            if 0 <= x0 < self.grid_size and 0 <= z0 < self.grid_size:
                cell = self.grid[x0, z0]
                cell.log_odds += self.log_odds_miss
                cell.misses += 1

            e2 = 2 * err
            if e2 > -dz:
                err -= dz
                x0 += sx
            if e2 < dx:
                err += dx
                z0 += sz

    def get_visualization(self) -> np.ndarray:
        """Create a visualization of the occupancy grid."""
        vis = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Draw grid cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.grid[i, j]
                if cell.log_odds > self.log_odds_threshold:
                    # Occupied - red
                    vis[i, j] = [0, 0, 255]
                elif cell.log_odds < -self.log_odds_threshold:
                    # Free - green
                    vis[i, j] = [0, 255, 0]
                else:
                    # Unknown - gray
                    vis[i, j] = [128, 128, 128]

        # Draw camera FOV lines if we have a current position
        if hasattr(self, 'current_pos') and hasattr(self, 'current_rot'):
            center_x, center_z = self.world_to_grid(self.current_pos[0], self.current_pos[2])
            
            # Calculate FOV cone endpoints
            angle_left = self.current_rot[1] - self.fov_horizontal/2
            angle_right = self.current_rot[1] + self.fov_horizontal/2
            
            # Convert angles to world coordinates
            left_x = self.current_pos[0] + self.max_range * np.sin(angle_left)
            left_z = self.current_pos[2] + self.max_range * np.cos(angle_left)
            right_x = self.current_pos[0] + self.max_range * np.sin(angle_right)
            right_z = self.current_pos[2] + self.max_range * np.cos(angle_right)
            
            # Convert to grid coordinates
            left_grid_x, left_grid_z = self.world_to_grid(left_x, left_z)
            right_grid_x, right_grid_z = self.world_to_grid(right_x, right_z)
            
            # Draw FOV lines
            cv2.line(vis, (center_z, center_x), (left_grid_z, left_grid_x), (255, 255, 0), 1)
            cv2.line(vis, (center_z, center_x), (right_grid_z, right_grid_x), (255, 255, 0), 1)
            
            # Draw current position
            cv2.circle(vis, (center_z, center_x), 3, (255, 255, 255), -1)
        
        return vis

    def get_probability_grid(self) -> np.ndarray:
        """Get the probability grid (0-1) where 1 means occupied."""
        prob_grid = np.zeros((self.grid_size, self.grid_size))
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.grid[i, j]
                # Convert log odds to probability
                prob = 1.0 / (1.0 + np.exp(-cell.log_odds))
                prob_grid[i, j] = prob
        
        return prob_grid

# Singleton instance
occupancy_grid = OccupancyGrid()
__all__ = ['occupancy_grid'] 