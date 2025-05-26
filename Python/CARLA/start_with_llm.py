import carla
import pygame
import numpy as np
import random
import cv2
import time
import threading
from LLMServiceClass import LLMService
from occupancy_grid_mapper import update_occupancy_grid, draw_occupancy_grid

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

# === Available Commands ===
AVAILABLE_COMMANDS = {
    "1": "Find a safe parking spot and stop",
    "2": "Follow the road ahead",
    "3": "Avoid obstacles and continue",
    "4": "Stop the vehicle",
    "5": "Turn around",
    "6": "Exit simulation"
}

class CarlaSimulation:
    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        # Initialize pygame
        pygame.init()
        self.font = pygame.font.SysFont("Arial", 18)
        self.display = pygame.display.set_mode((2400, 1200), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        
        # Initialize surfaces
        self.rgb_surface = None
        self.depth_surface = None
        self.segm_surface = None
        self.inst_surface = None
        self.segm_filtered_surface = None
        self.filtered_depth_surface = None
        self.segm_label_counts = {}
        self.segm_centroids = []
        self.segm_array = None
        self.depth_mode = 0
        self.frame_count = 0
        
        # Initialize vehicle and sensors
        self.setup_vehicle_and_sensors()
        
        # LLM processing variables
        self.current_instruction = None
        self.llm_output = None
        self.processing_llm = False
        self.command_thread = None

    def print_command_menu(self):
        print("\n=== Available Commands ===")
        for key, value in AVAILABLE_COMMANDS.items():
            print(f"{key}: {value}")
        print("========================")

    def handle_command_input(self):
        while True:
            self.print_command_menu()
            command = input("Enter command number: ").strip()
            
            if command == "6":  # Exit
                pygame.quit()
                exit()
            
            if command in AVAILABLE_COMMANDS:
                self.current_instruction = AVAILABLE_COMMANDS[command]
                print(f"\nExecuting command: {self.current_instruction}")
                
                # Prepare combined image for LLM
                combined_image = self.prepare_llm_image()
                if combined_image is not None:
                    self.processing_llm = True
                    self.llm_output = self.llm_service.process_image(combined_image, self.current_instruction)
                    self.processing_llm = False
                    print("\nLLM Response:", self.llm_output)
                else:
                    print("Error: Could not prepare image for LLM processing")
            else:
                print("Invalid command. Please try again.")

    def setup_vehicle_and_sensors(self):
        # Spawn vehicle
        vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.*'))
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Camera setup
        camera_transform = carla.Transform(carla.Location(x=4.0, z=1.5), carla.Rotation(pitch=0.0))
        self.cam_res = (800, 600)
        self.cam_fov = 90

        # Configure and spawn cameras
        self.rgb_cam = self.setup_camera('sensor.camera.rgb', camera_transform)
        self.depth_cam = self.setup_camera('sensor.camera.depth', camera_transform)
        self.segm_cam = self.setup_camera('sensor.camera.semantic_segmentation', camera_transform)
        self.inst_cam = self.setup_camera('sensor.camera.instance_segmentation', camera_transform)

    def setup_camera(self, sensor_type, transform):
        bp = self.blueprint_library.find(sensor_type)
        bp.set_attribute('image_size_x', str(self.cam_res[0]))
        bp.set_attribute('image_size_y', str(self.cam_res[1]))
        bp.set_attribute('fov', str(self.cam_fov))
        bp.set_attribute('sensor_tick', '0.05')
        return self.world.spawn_actor(bp, transform, attach_to=self.vehicle)

    def to_surface(self, image):
        img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
        img = img[:, :, ::-1]
        return pygame.surfarray.make_surface(img.swapaxes(0, 1))

    def process_rgb(self, image):
        self.rgb_surface = self.to_surface(image)
        # Store the RGB image for LLM processing
        self.current_rgb_image = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]

    def prepare_llm_image(self):
        """Combine depth and occupancy grid images for LLM processing"""
        if not hasattr(self, 'current_rgb_image') or self.filtered_depth_surface is None:
            return None

        # Get depth image from surface
        depth_array = pygame.surfarray.array3d(self.filtered_depth_surface)
        depth_array = np.transpose(depth_array, (1, 0, 2))  # Fix orientation

        # Get occupancy grid image
        occ_surface = draw_occupancy_grid()
        occ_array = pygame.surfarray.array3d(occ_surface)
        occ_array = np.transpose(occ_array, (1, 0, 2))  # Fix orientation

        # Resize occupancy grid to match depth image size
        occ_array = cv2.resize(occ_array, (depth_array.shape[1], depth_array.shape[0]))

        # Create a combined image (depth on top, occupancy grid on bottom)
        combined_height = depth_array.shape[0] * 2
        combined_width = depth_array.shape[1]
        combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Place depth image on top
        combined_image[:depth_array.shape[0], :] = depth_array
        # Place occupancy grid on bottom
        combined_image[depth_array.shape[0]:, :] = occ_array

        return combined_image

    def process_depth(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        rgb_depth = array[:, :, :3].copy()

        r = rgb_depth[:, :, 0].astype(np.float32)
        g = rgb_depth[:, :, 1].astype(np.float32)
        b = rgb_depth[:, :, 2].astype(np.float32)
        norm_depth = (r + g / 256.0 + b / (256.0 * 256.0)) / 256.0
        depth_meters = norm_depth * 1000.0

        valid_mask = (depth_meters > 0.01) & (depth_meters < 1000)
        if self.segm_array is not None:
            obstacle_mask = np.isin(self.segm_array, OBSTACLE_CLASS_IDS)
            combined_mask = valid_mask & obstacle_mask
        else:
            combined_mask = valid_mask

        depth_clipped = np.clip(depth_meters, 0, 80)
        norm = ((80.0 - depth_clipped) / 80.0 * 255).astype(np.uint8)
        norm[~combined_mask] = 0
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        self.filtered_depth_surface = pygame.surfarray.make_surface(heatmap.swapaxes(0, 1))

        if self.segm_array is not None:
            segm_mask = np.isin(self.segm_array, OBSTACLE_CLASS_IDS)
            vehicle_transform = self.vehicle.get_transform()
            update_occupancy_grid(rgb_depth, segm_mask, vehicle_transform, fov_deg=90.0)

    def process_segm(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        labels = array[:, :, 2]
        self.segm_array = labels.copy()

        unique, counts = np.unique(labels, return_counts=True)
        self.segm_label_counts = dict(zip(unique, counts))

        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for class_id in SEMANTIC_LABELS:
            mask[labels == class_id] = 255

        self.segm_filtered_surface = pygame.surfarray.make_surface(np.stack([mask]*3, axis=2).swapaxes(0, 1))

        self.segm_centroids = []
        for label_id in SEMANTIC_LABELS:
            bin_mask = np.uint8(labels == label_id)
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        self.segm_centroids.append((cx, cy, label_id))

        image.convert(carla.ColorConverter.CityScapesPalette)
        self.segm_surface = self.to_surface(image)

    def process_inst(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        self.inst_surface = self.to_surface(image)

    def apply_llm_control(self):
        if self.llm_output and 'actions' in self.llm_output:
            for action in self.llm_output['actions']:
                if action == "slow_down":
                    self.vehicle.apply_control(carla.VehicleControl(throttle=0.1, brake=0.5))
                elif action == "stop":
                    self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                elif action == "turn_left":
                    self.vehicle.apply_control(carla.VehicleControl(throttle=0.1, steer=-0.5))
                elif action == "turn_right":
                    self.vehicle.apply_control(carla.VehicleControl(throttle=0.1, steer=0.5))
                elif action == "forward":
                    self.vehicle.apply_control(carla.VehicleControl(throttle=0.5))
        else:
            # Default control if no LLM output
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.1))

    def run(self):
        # Start command input thread
        self.command_thread = threading.Thread(target=self.handle_command_input, daemon=True)
        self.command_thread.start()

        # Set up sensor callbacks
        self.rgb_cam.listen(self.process_rgb)
        self.depth_cam.listen(self.process_depth)
        self.segm_cam.listen(self.process_segm)
        self.inst_cam.listen(self.process_inst)

        try:
            while True:
                fps = self.clock.get_fps()
                self.clock.tick(30)
                self.frame_count += 1

                # Apply controls based on LLM output
                self.apply_llm_control()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_l:
                            self.depth_mode = (self.depth_mode + 1) % 4
                            print(f"[TOGGLE] Depth mode: {self.depth_mode}")

                self.display.fill((0, 0, 0))

                # Draw all surfaces
                if self.rgb_surface: self.display.blit(self.rgb_surface, (0, 0))
                if self.depth_surface: self.display.blit(self.depth_surface, (800, 0))
                if self.segm_surface:
                    self.display.blit(self.segm_surface, (0, 600))
                    for cx, cy, label_id in self.segm_centroids:
                        label = SEMANTIC_LABELS.get(label_id, f"Class {label_id}")
                        text = self.font.render(label, True, (255, 255, 0))
                        self.display.blit(text, (cx, 600 + cy))
                if self.segm_filtered_surface:
                    self.display.blit(self.segm_filtered_surface, (800, 600))
                if self.filtered_depth_surface:
                    self.display.blit(self.filtered_depth_surface, (1600, 0))

                # Draw occupancy map
                occ_surface = draw_occupancy_grid()
                occ_surface = pygame.transform.scale(occ_surface, (800, 600))
                self.display.blit(occ_surface, (1600, 600))

                # Draw LLM output if available
                if self.llm_output:
                    y_offset = 10
                    for key, value in self.llm_output.items():
                        text = self.font.render(f"{key}: {value}", True, (255, 255, 255))
                        self.display.blit(text, (10, y_offset))
                        y_offset += 20

                # Draw overlay
                overlay = [
                    f"FPS: {fps:.1f}",
                    f"Depth mode: {self.depth_mode}",
                    f"RGB: {self.cam_res[0]}x{self.cam_res[1]}, FOV: {self.cam_fov}",
                    f"Depth: {self.cam_res[0]}x{self.cam_res[1]}, FOV: {self.cam_fov}",
                    "Press [L] to toggle depth mode",
                    f"Current command: {self.current_instruction if self.current_instruction else 'None'}",
                    "Processing LLM..." if self.processing_llm else ""
                ]
                for i, line in enumerate(overlay):
                    text = self.font.render(line, True, (255, 255, 255))
                    self.display.blit(text, (10, 10 + i * 20))

                pygame.display.flip()

        except KeyboardInterrupt:
            print("Shutting down.")
        finally:
            self.cleanup()

    def cleanup(self):
        for sensor in [self.rgb_cam, self.depth_cam, self.segm_cam, self.inst_cam]:
            sensor.stop()
            sensor.destroy()
        self.vehicle.destroy()
        pygame.quit()

def main():
    # Initialize LLM service
    llm_service = LLMService(api_key="your-api-key-here")
    
    # Create and run simulation
    sim = CarlaSimulation(llm_service)
    sim.run()

if __name__ == '__main__':
    main() 