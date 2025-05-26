import cv2
import numpy as np
from depth_estimator import DepthEstimator
from object_segmenter import ObjectSegmenter

class SimpleMapper:
    def __init__(self, grid_size=0.2, map_dim=(100, 100), camera_height=1.6):
        self.grid_size = grid_size
        self.map_w, self.map_h = map_dim
        self.camera_height = camera_height
        
        # Camera intrinsics (for 256x256 image)
        image_width = 256
        image_height = 256
        sensor_width = 36  # mm
        sensor_height = 24  # mm
        focal_length = 50  # mm

        self.fx = (image_width * focal_length) / sensor_width
        self.fy = (image_height * focal_length) / sensor_height
        self.cx = image_width / 2
        self.cy = image_height / 2
        
        # Initialize occupancy grid
        self.occupancy_grid = np.zeros((self.map_h, self.map_w), dtype=np.uint8)

    def update_from_depth(self, depth_map, detections):
        """Update map using depth information and detections"""
        # Clear previous grid
        self.occupancy_grid.fill(0)
        
        center_x = self.map_w // 2
        center_y = self.map_h // 2
        
        # Process each detection
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # Get depth at bottom center of bounding box
            bottom_center_x = int((x1 + x2) / 2)
            bottom_center_y = int(y2)
            
            if 0 <= bottom_center_x < depth_map.shape[1] and 0 <= bottom_center_y < depth_map.shape[0]:
                # Get depth value
                depth = depth_map[bottom_center_y, bottom_center_x]
                
                # Project to ground plane
                z = depth * self.camera_height
                x = (bottom_center_x - self.cx) * z / self.fx
                
                # Convert to grid coordinates
                grid_x = int(x / self.grid_size) + center_x
                grid_y = int(z / self.grid_size) + center_y
                
                # Check if point is within grid bounds
                if 0 <= grid_x < self.map_w and 0 <= grid_y < self.map_h:
                    # Draw a small circle at the projected point
                    cv2.circle(self.occupancy_grid, (grid_x, grid_y), 2, 255, -1)

    def get_visualization(self):
        """Get visualization of the occupancy grid"""
        # Convert to 3-channel image
        vis = np.stack([self.occupancy_grid]*3, axis=-1)
        
        # Resize for better visualization
        vis = cv2.resize(vis, (300, 300), interpolation=cv2.INTER_NEAREST)
        
        # Apply colormap
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        
        # Add grid lines
        grid_spacing = 10
        for i in range(0, vis.shape[0], grid_spacing):
            cv2.line(vis, (0, i), (vis.shape[1], i), (50, 50, 50), 1)
        for i in range(0, vis.shape[1], grid_spacing):
            cv2.line(vis, (i, 0), (i, vis.shape[0]), (50, 50, 50), 1)
            
        return vis

def process_frame(frame):
    """Process a single frame with depth estimation and mapping"""
    # Initialize models
    depth_estimator = DepthEstimator(model_type="MiDaS_small")
    object_segmenter = ObjectSegmenter(model_path="yolov8s-seg.pt")
    mapper = SimpleMapper()
    
    # 1. Get depth map
    depth_map, depth_error = depth_estimator.estimate_depth(frame)
    if depth_error:
        print(f"Depth estimation error: {depth_error}")
        return None, None, None, None
    
    # 2. Get object detections
    detections, det_error = object_segmenter.detect_and_segment(frame)
    if det_error:
        print(f"Detection error: {det_error}")
        return None, None, None, None
    
    # 3. Update map
    mapper.update_from_depth(depth_map, detections)
    
    # 4. Get visualizations
    depth_vis = depth_estimator.get_visualization(depth_map)
    det_vis, seg_vis = object_segmenter.get_visualization(frame, detections)
    map_vis = mapper.get_visualization()
    
    return depth_vis, det_vis, seg_vis, map_vis

def main():
    # Initialize video capture (0 for webcam, or provide video file path)
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process frame
        depth_vis, det_vis, seg_vis, map_vis = process_frame(frame)
        
        if depth_vis is not None:
            # Display results
            cv2.imshow("Depth Map", depth_vis)
            cv2.imshow("Object Detection", det_vis)
            cv2.imshow("Instance Segmentation", seg_vis)
            cv2.imshow("Top-Down View", map_vis)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 