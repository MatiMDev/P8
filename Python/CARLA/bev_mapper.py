import numpy as np
import cv2
import pygame

class OccupancyGridMapper:
    def __init__(self, grid_size_m=100, resolution=0.2):
        self.grid_size_m = grid_size_m
        self.resolution = resolution
        self.grid_size_px = int(grid_size_m / resolution)
        self.grid = np.zeros((self.grid_size_px, self.grid_size_px), dtype=np.uint8)

    def world_to_grid(self, x, y, ego_x, ego_y):
        dx = x - ego_x
        dy = y - ego_y
        ix = int(self.grid_size_px / 2 + dx / self.resolution)
        iy = int(self.grid_size_px / 2 - dy / self.resolution)
        return ix, iy

    def reset(self):
        self.grid.fill(0)

    def update(self, points_world, ego_location):
        for x, y, _ in points_world:
            gx, gy = self.world_to_grid(x, y, ego_location.x, ego_location.y)
            if 0 <= gx < self.grid_size_px and 0 <= gy < self.grid_size_px:
                self.grid[gy, gx] = 255

    def get_surface(self, scale=400):
        vis = cv2.cvtColor(self.grid, cv2.COLOR_GRAY2RGB)
        vis = cv2.resize(vis, (scale, scale), interpolation=cv2.INTER_NEAREST)
        return pygame.surfarray.make_surface(vis.swapaxes(0, 1))


def depth_to_local_point_cloud(depth_image, segm_image, cam_fov, only_obstacles=True, obstacle_ids=None):
    height, width = depth_image.height, depth_image.width
    array = np.frombuffer(depth_image.raw_data, dtype=np.uint8).reshape((height, width, 4))
    segm = np.frombuffer(segm_image.raw_data, dtype=np.uint8).reshape((height, width, 4))[:, :, 2]

    red = array[:, :, 0].astype(np.float32)
    depth_m = red / 255.0 * 1000.0

    mask = (depth_m > 0.01) & (depth_m < 1000.0)
    if only_obstacles and obstacle_ids is not None:
        mask &= np.isin(segm, obstacle_ids)

    fx = fy = width / (2.0 * np.tan(np.radians(cam_fov) / 2.0))
    cx = width / 2.0
    cy = height / 2.0

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m

    x = x[mask]
    y = y[mask]
    z = z[mask]
    return np.stack([x, y, z], axis=1)


def transform_points(points, transform):
    matrix = np.array(transform.get_matrix())
    world_points = []
    for pt in points:
        local = np.array([pt[0], pt[1], pt[2], 1.0])
        world = matrix @ local
        world_points.append((world[0], world[1], world[2]))
    return np.array(world_points)
