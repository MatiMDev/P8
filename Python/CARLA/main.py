import glob
import os
import sys
import numpy as np
import carla
import pygame
import time
import cv2

# ==== Add CARLA egg ====
try:
    sys.path.append(glob.glob('../../dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==== Init CARLA ====
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprints = world.get_blueprint_library()

# ==== Pick first available vehicle ====
vehicle_bp = None
vehicle_list = blueprints.filter('vehicle.*')
if vehicle_list:
    vehicle_bp = vehicle_list[0]
    print(f"Using vehicle: {vehicle_bp.id}")
else:
    raise RuntimeError("No vehicles available in your CARLA install!")

# ==== Spawn vehicle ====
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# ==== Sensor settings ====
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
WIDTH, HEIGHT = 640, 480

# ==== RGB camera ====
rgb_bp = blueprints.find('sensor.camera.rgb')
rgb_bp.set_attribute('image_size_x', f"{WIDTH}")
rgb_bp.set_attribute('image_size_y', f"{HEIGHT}")
rgb_bp.set_attribute('fov', '90')
rgb_cam = world.spawn_actor(rgb_bp, camera_transform, attach_to=vehicle)

# ==== Depth camera ====
depth_bp = blueprints.find('sensor.camera.depth')
depth_bp.set_attribute('image_size_x', f"{WIDTH}")
depth_bp.set_attribute('image_size_y', f"{HEIGHT}")
depth_bp.set_attribute('fov', '90')
depth_cam = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)

# ==== Pygame viewer ====
pygame.init()
display = pygame.display.set_mode((WIDTH * 2, HEIGHT))
pygame.display.set_caption("RGB (Left) + Depth (Right)")
clock = pygame.time.Clock()

rgb_surface = None
depth_surface = None

# ==== Decode depth ====
def decode_carla_depth(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    normalized = (array[:, :, 0] +
                  array[:, :, 1] * 256 +
                  array[:, :, 2] * 256 * 256) / float(256**3 - 1)
    return 1000.0 * normalized  # meters

# ==== Sensor callbacks ====
def on_rgb(image):
    global rgb_surface
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    array = array[:, :, ::-1]  # BGR to RGB
    rgb_surface = pygame.surfarray.make_surface(np.transpose(array, (1, 0, 2)))

def on_depth(image):
    global depth_surface
    depth = decode_carla_depth(image)  # shape: (H, W), values in meters

    # Visualization: color map for easier interpretation
    vis = np.clip(depth / 50.0 * 255.0, 0, 255).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    vis_color = cv2.cvtColor(vis_color, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR

    depth_surface = pygame.surfarray.make_surface(np.transpose(vis_color, (1, 0, 2)))

# ==== Start streaming ====
rgb_cam.listen(on_rgb)
depth_cam.listen(on_depth)

# ==== Display loop ====
try:
    print("Press ESC or close window to exit.")
    while True:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                raise KeyboardInterrupt

        if rgb_surface and depth_surface:
            display.blit(rgb_surface, (0, 0))
            display.blit(depth_surface, (WIDTH, 0))
            pygame.display.flip()

except KeyboardInterrupt:
    print("Exiting...")

finally:
    rgb_cam.stop()
    depth_cam.stop()
    rgb_cam.destroy()
    depth_cam.destroy()
    vehicle.destroy()
    pygame.quit()
