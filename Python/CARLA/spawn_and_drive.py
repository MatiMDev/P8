import carla
import random
import time

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*mini.cooper')[0]

    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Spawned vehicle: {vehicle.type_id}")

    # Set up RGB camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')

    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # hood mount
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Set up display
    camera.listen(lambda image: image.save_to_disk('_out/frame_%06d.png' % image.frame))

    # Drive slowly forward
    vehicle.apply_control(carla.VehicleControl(throttle=0.3))

    try:
        print("Running for 20 seconds...")
        time.sleep(20)

    finally:
        print("Cleaning up")
        camera.stop()
        camera.destroy()
        vehicle.destroy()

if __name__ == '__main__':
    main()
