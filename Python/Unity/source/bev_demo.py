import cv2
import numpy as np
from mapper import mapper

def create_sample_depth_map():
    # Create a sample depth map (256x256)
    depth_map = np.ones((256, 256), dtype=np.float32) * 10.0  # 10 meters default depth
    
    # Add some sample obstacles
    # A wall in front
    depth_map[100:150, 100:150] = 5.0
    
    # Some random obstacles
    depth_map[50:70, 180:200] = 3.0
    depth_map[180:200, 50:70] = 4.0
    
    return depth_map

def main():
    print("🚗 Starting BEV Mapping Demo")
    print("Press 'q' to quit")
    
    # Sample camera position and rotation
    position = [0.0, 0.0, 0.0]  # x, y, z in meters
    rotation = [0.0, 0.0, 0.0]  # roll, pitch, yaw in degrees
    
    while True:
        # Create sample depth map
        depth_map = create_sample_depth_map()
        
        # Update the BEV map
        mapper.update_from_depth(depth_map, position, rotation)
        
        # Get visualization
        bev_vis = mapper.get_visualization()
        
        # Display the BEV map
        cv2.imshow("Bird's Eye View Map", bev_vis)
        
        # Simulate some movement
        position[0] += 0.1  # Move forward
        rotation[2] += 1.0  # Rotate slightly
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 