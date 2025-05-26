import carla
import pygame
import numpy as np
import random
import cv2

from bev_mapper import OccupancyGridMapper, depth_to_local_point_cloud, transform_points

mapper = OccupancyGridMapper(grid_size_m=100, resolution=0.2)

# === Tesla-style important labels ===
SEMANTIC_LABELS = {
    6: "Pole",
    7: "TrafficLight",
    12: "Pedestrian",
    14: "Car"
}

# === Obstacle classes to include in depth heatmap ===
OBSTACLE_CLASS_IDS = [
    3,  # Building
    4,  # Wall
    5,  # Fence
    6,  # Pole
    7,  # TrafficLight
    8,  # TrafficSign
    9,
    12, # Pedestrian
    13, # Rider
    14, # Car
    15, # Truck
    16, # Bus
    17, # Train
    18, # Motorcycle
    19, # Bicycle
    20, # Static
    21, # Dynamic
    26, # Bridge
    28  # GuardRail
]


def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    camera_transform = carla.Transform(carla.Location(x=6.0, z=2.5), carla.Rotation(pitch=-5.0))
    cam_res = (800, 600)
    cam_fov = 90

    def configure_camera(bp, res, fov):
        bp.set_attribute('image_size_x', str(res[0]))
        bp.set_attribute('image_size_y', str(res[1]))
        bp.set_attribute('fov', str(fov))
        bp.set_attribute('sensor_tick', '0.05')

    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    depth_bp = blueprint_library.find('sensor.camera.depth')
    #filtered_depth_bp = blueprint_library.find('sensor.camera.depth')
    segm_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    inst_bp = blueprint_library.find('sensor.camera.instance_segmentation')

    for bp in [rgb_bp, depth_bp, segm_bp, inst_bp]:
        configure_camera(bp, cam_res, cam_fov)

    rgb_cam = world.spawn_actor(rgb_bp, camera_transform, attach_to=vehicle)
    depth_cam = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
    #filtered_depth_cam = world.spawn_actor(filtered_depth_bp, camera_transform, attach_to=vehicle)
    segm_cam = world.spawn_actor(segm_bp, camera_transform, attach_to=vehicle)
    inst_cam = world.spawn_actor(inst_bp, camera_transform, attach_to=vehicle)

    pygame.init()
    font = pygame.font.SysFont("Arial", 18)
    display = pygame.display.set_mode((2400, 1200), pygame.HWSURFACE | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    rgb_surface = None
    segm_image = None
    depth_surface = None
    segm_surface = None
    inst_surface = None
    segm_filtered_surface = None
    filtered_depth_surface = None
    segm_label_counts = {}
    segm_centroids = []
    segm_array = None
    depth_mode = 0
    frame_count = 0

    def get_depth_mode_name(mode):
        return ["Logarithmic", "Gray Scale", "Raw", "Decoded Depth (m)"][mode]

    def to_surface(image):
        img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
        img = img[:, :, ::-1]
        return pygame.surfarray.make_surface(img.swapaxes(0, 1))

    def process_rgb(image):
        nonlocal rgb_surface
        rgb_surface = to_surface(image)

    def process_depth(image):
        nonlocal filtered_depth_surface

        if segm_cam is None or segm_image is None:
            return  # wait for both depth and segmentation

        # === Generate point cloud from depth + segmentation
        local_points = depth_to_local_point_cloud(image, segm_image, cam_fov=90.0, only_obstacles=True, obstacle_ids=OBSTACLE_CLASS_IDS)

        # === Transform points to world frame
        camera_transform = depth_cam.get_transform()
        world_points = transform_points(local_points, camera_transform)

        # === Update occupancy grid
        mapper.update(world_points, vehicle.get_location())

        # === Optional: display filtered heatmap (if needed)
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        red = array[:, :, 0].astype(np.float32)
        norm_depth = red / 255.0
        depth_meters = norm_depth * 1000.0
        norm = ((80.0 - np.clip(depth_meters, 0, 80)) / 80.0 * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        filtered_depth_surface = pygame.surfarray.make_surface(heatmap.swapaxes(0, 1))


    def process_heatmap(image):
        nonlocal filtered_depth_surface

        image.convert(carla.ColorConverter.Depth)
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        red = array[:, :, 0].astype(np.float32)
        norm_depth = red / 255.0
        depth_meters = norm_depth * 1000.0
        valid_mask = (depth_meters > 0.01) & (depth_meters < 1000)

        if segm_array is not None:
            obstacle_mask = np.isin(segm_array, OBSTACLE_CLASS_IDS)
            combined_mask = valid_mask & obstacle_mask
        else:
            combined_mask = valid_mask

        depth_clipped = np.clip(depth_meters, 0, 80)
        norm = ((80.0 - depth_clipped) / 80.0 * 255).astype(np.uint8)
        norm[~combined_mask] = 0

        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        filtered_depth_surface = pygame.surfarray.make_surface(heatmap.swapaxes(0, 1))

    def process_segm(image):
        nonlocal segm_image
        segm_image = image
        nonlocal segm_surface, segm_filtered_surface, segm_label_counts, segm_centroids, segm_array
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        labels = array[:, :, 2]
        segm_array = labels.copy()

        unique, counts = np.unique(labels, return_counts=True)
        segm_label_counts = dict(zip(unique, counts))

        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for class_id in SEMANTIC_LABELS:
            mask[labels == class_id] = 255

        segm_filtered_surface = pygame.surfarray.make_surface(np.stack([mask]*3, axis=2).swapaxes(0, 1))

        segm_centroids = []
        for label_id in SEMANTIC_LABELS:
            bin_mask = np.uint8(labels == label_id)
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        segm_centroids.append((cx, cy, label_id))

        image.convert(carla.ColorConverter.CityScapesPalette)
        segm_surface = to_surface(image)

    def process_inst(image):
        nonlocal inst_surface
        image.convert(carla.ColorConverter.CityScapesPalette)
        inst_surface = to_surface(image)

    rgb_cam.listen(process_rgb)
    depth_cam.listen(process_depth)
    segm_cam.listen(process_segm)
    inst_cam.listen(process_inst)

    try:
        while True:
            fps = clock.get_fps()
            clock.tick(30)
            frame_count += 1

            vehicle.apply_control(carla.VehicleControl(throttle=0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                    depth_mode = (depth_mode + 1) % 4
                    print(f"[TOGGLE] Depth mode: {depth_mode} ({get_depth_mode_name(depth_mode)})")

            display.fill((0, 0, 0))

            if rgb_surface: display.blit(rgb_surface, (0, 0))
            if depth_surface: display.blit(depth_surface, (800, 0))
            if segm_surface:
                display.blit(segm_surface, (0, 600))
                for cx, cy, label_id in segm_centroids:
                    label = SEMANTIC_LABELS.get(label_id, f"Class {label_id}")
                    text = font.render(label, True, (255, 255, 0))
                    display.blit(text, (cx, 600 + cy))
            if segm_filtered_surface:
                display.blit(segm_filtered_surface, (800, 600))
            if filtered_depth_surface:
                display.blit(filtered_depth_surface, (1600, 0))

            # === Display Occupancy Map ===
            grid_surface = mapper.get_surface(scale=400)
            display.blit(grid_surface, (1200, 800))  # Adjust as needed



            overlay = [
                f"FPS: {fps:.1f}",
                f"Depth mode: {get_depth_mode_name(depth_mode)}",
                f"RGB: {cam_res[0]}x{cam_res[1]}, FOV: {cam_fov}",
                f"Depth: {cam_res[0]}x{cam_res[1]}, FOV: {cam_fov}",
                "Press [L] to toggle depth mode"
            ]
            for i, line in enumerate(overlay):
                text = font.render(line, True, (255, 255, 255))
                display.blit(text, (10, 10 + i * 20))

            pygame.display.flip()

    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        for sensor in [rgb_cam, depth_cam, segm_cam, inst_cam]:
            sensor.stop()
            sensor.destroy()
        vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
