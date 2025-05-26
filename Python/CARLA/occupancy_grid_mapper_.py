import numpy as np
import pygame
import cv2

# === Grid Config ===
GRID_WIDTH = 800
GRID_HEIGHT = 800
GRID_RESOLUTION = 0.5  # meters per cell

# === Persistent Occupancy Grid ===
occupancy_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)

# Store last vehicle pose for FOV triangle drawing
last_vehicle_pos = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
last_yaw_deg = 0.0


def update_occupancy_grid(rgb_depth, segm_mask, vehicle_transform, fov_deg=90.0):
    global occupancy_grid, last_vehicle_pos, last_yaw_deg

    h, w, _ = rgb_depth.shape

    # === Camera intrinsics ===
    fov = np.radians(fov_deg)
    fx = w / (2 * np.tan(fov / 2))
    fy = fx  # square pixels assumed
    cx = w / 2.0
    cy = h / 2.0

    # === Decode depth ===
    r = rgb_depth[:, :, 0].astype(np.float32)
    g = rgb_depth[:, :, 1].astype(np.float32)
    b = rgb_depth[:, :, 2].astype(np.float32)
    depth_normalized = (r + g / 256.0 + b / (256.0 * 256.0)) / 256.0
    depth_meters = np.clip(depth_normalized * 1000.0, 0.01, 80.0)

    # === Sample pixels ===
    step = 4  # Instead of 1
    ys, xs = np.mgrid[int(h * 0.4):h:step, 0:w:step]  # Only bottom half of image


    xs = xs.ravel()
    ys = ys.ravel()

    zs = depth_meters[ys, xs]
    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy

    # === Optional mask
    if segm_mask is not None:
        if segm_mask.dtype != np.bool_:
            segm_mask = segm_mask.astype(bool)
        mask = segm_mask[ys, xs]
    else:
        mask = np.ones_like(zs, dtype=bool)

    # === World coordinates
    yaw = np.radians(vehicle_transform.rotation.yaw)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    x_world = vehicle_transform.location.x + zs * cos_yaw - x_cam * sin_yaw
    y_world = vehicle_transform.location.y + zs * sin_yaw + x_cam * cos_yaw

    # === Grid coordinates
    grid_x = np.round(GRID_WIDTH / 2 + x_world / GRID_RESOLUTION).astype(int)
    grid_y = np.round(GRID_HEIGHT / 2 - y_world / GRID_RESOLUTION).astype(int)

    # === Vehicle grid position (for ray origin)
    vx = vehicle_transform.location.x
    vy = vehicle_transform.location.y
    vehicle_grid_x = int(GRID_WIDTH / 2 + vx / GRID_RESOLUTION)
    vehicle_grid_y = int(GRID_HEIGHT / 2 - vy / GRID_RESOLUTION)

    # === Raytrace and update grid
    for gx, gy, valid_flag in zip(grid_x, grid_y, mask):
        if not valid_flag:
            continue
        # Trace ray from vehicle → (gx, gy)
        line_points = bresenham(vehicle_grid_x, vehicle_grid_y, gx, gy)
        for lx, ly in line_points[:-1]:  # free space
            if 0 <= lx < GRID_WIDTH and 0 <= ly < GRID_HEIGHT:
                occupancy_grid[ly, lx] = 0
        # Obstacle cell
        if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
            occupancy_grid[gy, gx] = 255

    # === Store pose for drawing
    last_vehicle_pos = (vehicle_grid_x, vehicle_grid_y)
    last_yaw_deg = vehicle_transform.rotation.yaw


def draw_occupancy_grid():
    global last_vehicle_pos, last_yaw_deg

    surface = np.stack([occupancy_grid] * 3, axis=-1)

    x, y = last_vehicle_pos
    yaw = np.radians(last_yaw_deg)

    # Triangle: front = 20m, width = 10m
    front = np.array([x + 20 * np.cos(yaw), y - 20 * np.sin(yaw)])
    left  = np.array([x + 10 * np.cos(yaw + np.pi/2), y - 10 * np.sin(yaw + np.pi/2)])
    right = np.array([x + 10 * np.cos(yaw - np.pi/2), y - 10 * np.sin(yaw - np.pi/2)])
    triangle = np.array([left, right, front], dtype=np.int32)

    cv2.polylines(surface, [triangle], isClosed=True, color=(0, 255, 0), thickness=2)
    return pygame.surfarray.make_surface(surface.swapaxes(0, 1))


def bresenham(x0, y0, x1, y1):
    """2D Bresenham's line algorithm"""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points
