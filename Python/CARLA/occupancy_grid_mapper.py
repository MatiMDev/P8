import numpy as np
import pygame
import cv2

# === Grid Config ===
GRID_WIDTH = 800
GRID_HEIGHT = 800
GRID_RESOLUTION = 0.5  # meters per cell

# === Log-odds Probabilistic Grid ===
log_odds_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
log_odds_occ = np.log(9)      # p=0.9
log_odds_free = np.log(1 / 9) # p=0.1
log_odds_min = -5
log_odds_max = 5

last_vehicle_pos = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
last_yaw_deg = 0.0
trajectory = []

def update_occupancy_grid(rgb_depth, segm_mask, vehicle_transform, fov_deg=90.0):
    global log_odds_grid, last_vehicle_pos, last_yaw_deg, trajectory

    h, w, _ = rgb_depth.shape
    fov = np.radians(fov_deg)
    fx = w / (2 * np.tan(fov / 2))
    fy = fx
    cx = w / 2.0
    cy = h / 2.0

    # Decode depth
    r = rgb_depth[:, :, 0].astype(np.float32)
    g = rgb_depth[:, :, 1].astype(np.float32)
    b = rgb_depth[:, :, 2].astype(np.float32)
    depth = ((r + g / 256 + b / (256 ** 2)) / 256.0) * 1000.0
    depth = np.clip(depth, 0.01, 80.0)

    # Sample image points
    step = 4
    ys, xs = np.mgrid[int(h * 0.4):h:step, 0:w:step]
    ys = ys.ravel()
    xs = xs.ravel()
    zs = depth[ys, xs]

    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy

    pitch = np.radians(vehicle_transform.rotation.pitch)
    yaw = np.radians(vehicle_transform.rotation.yaw)

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)

    y_adj = y_cam * cos_pitch - zs * sin_pitch
    z_adj = y_cam * sin_pitch + zs * cos_pitch

    vx_world = vehicle_transform.location.x
    vy_world = vehicle_transform.location.y

    x_world = vx_world + z_adj * cos_yaw - x_cam * sin_yaw
    y_world = vy_world + z_adj * sin_yaw + x_cam * cos_yaw

    grid_x = np.round(GRID_WIDTH / 2 + x_world / GRID_RESOLUTION).astype(int)
    grid_y = np.round(GRID_HEIGHT / 2 - y_world / GRID_RESOLUTION).astype(int)

    vehicle_grid_x = int(GRID_WIDTH / 2 + vx_world / GRID_RESOLUTION)
    vehicle_grid_y = int(GRID_HEIGHT / 2 - vy_world / GRID_RESOLUTION)
    trajectory.append((vehicle_grid_x, vehicle_grid_y))

    if segm_mask is not None:
        mask = segm_mask[ys, xs].astype(bool)
    else:
        mask = np.ones_like(zs, dtype=bool)

    for gx, gy, valid in zip(grid_x, grid_y, mask):
        if not valid:
            continue
        for fx, fy in bresenham(vehicle_grid_x, vehicle_grid_y, gx, gy)[:-1]:
            if 0 <= fx < GRID_WIDTH and 0 <= fy < GRID_HEIGHT:
                log_odds_grid[fy, fx] += log_odds_free
                log_odds_grid[fy, fx] = np.clip(log_odds_grid[fy, fx], log_odds_min, log_odds_max)
        if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
            log_odds_grid[gy, gx] += log_odds_occ
            log_odds_grid[gy, gx] = np.clip(log_odds_grid[gy, gx], log_odds_min, log_odds_max)

    last_vehicle_pos = (vehicle_grid_x, vehicle_grid_y)
    last_yaw_deg = vehicle_transform.rotation.yaw

def draw_occupancy_grid():
    prob = 1 - 1 / (1 + np.exp(log_odds_grid))
    img = np.uint8(prob * 255)
    surface = np.stack([img] * 3, axis=-1)

    for gx, gy in trajectory:
        if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
            surface[gy, gx] = [0, 255, 255]  # cyan

    x, y = last_vehicle_pos
    yaw = np.radians(last_yaw_deg)
    front = np.array([x + 20 * np.cos(yaw), y - 20 * np.sin(yaw)])
    left  = np.array([x + 10 * np.cos(yaw + np.pi / 2), y - 10 * np.sin(yaw + np.pi / 2)])
    right = np.array([x + 10 * np.cos(yaw - np.pi / 2), y - 10 * np.sin(yaw - np.pi / 2)])
    triangle = np.array([left, right, front], dtype=np.int32)

    cv2.polylines(surface, [triangle], isClosed=True, color=(0, 255, 0), thickness=2)
    return pygame.surfarray.make_surface(surface.swapaxes(0, 1))

def bresenham(x0, y0, x1, y1):
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
