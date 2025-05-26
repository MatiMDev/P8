from flask import Flask, request
import logging
import io
import traceback
import numpy as np
import cv2
from PIL import Image
from depth_estimator import DepthEstimator

# Disable Flask logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Initialize Flask app
app = Flask(__name__)

# Initialize depth estimator
depth_estimator = DepthEstimator(model_type="MiDaS_small")

# Global state
state = {
    "latest_frame": None,
    "latest_depth": None,
    "latest_depth_vis": None,
    "latest_projection": None
}

def project_depth_to_ground(depth_map):
    """Project depth onto ground plane with obstacles in black and free space in solid white."""
    # Create a white image (free space)
    projection = np.ones((300, 300), dtype=np.uint8) * 255
    
    h, w = depth_map.shape
    obstacle_points = [[] for _ in range(300)]  # Store y for each x
    
    # Sample every pixel in x, every 4th in y
    for y in range(0, h, 4):
        for x in range(0, w):
            depth = depth_map[y, x]
            if depth > 0.1:
                proj_x = int(x * 300 / w)
                proj_y = int(depth * 300)
                if 0 <= proj_x < 300 and 0 <= proj_y < 300:
                    obstacle_points[proj_x].append(proj_y)
    
    # Find the first obstacle for each column (or None)
    first_obstacle_y = [min(pts) if pts else None for pts in obstacle_points]
    
    # Fill missing columns by nearest neighbor
    last_valid = None
    for x in range(300):
        if first_obstacle_y[x] is not None:
            last_valid = first_obstacle_y[x]
        else:
            # Look ahead for the next valid
            next_valid = None
            for x2 in range(x+1, 300):
                if first_obstacle_y[x2] is not None:
                    next_valid = first_obstacle_y[x2]
                    break
            # Choose the closer of last_valid and next_valid
            if last_valid is not None and next_valid is not None:
                if abs(x - (x-1)) <= abs(x2 - x):
                    first_obstacle_y[x] = last_valid
                else:
                    first_obstacle_y[x] = next_valid
            elif last_valid is not None:
                first_obstacle_y[x] = last_valid
            elif next_valid is not None:
                first_obstacle_y[x] = next_valid
            else:
                first_obstacle_y[x] = 299  # Default to bottom
    
    # Now, for each column, fill below the first obstacle as black
    for x in range(300):
        y0 = first_obstacle_y[x]
        if y0 is not None:
            projection[y0:, x] = 0  # Fill from first obstacle downwards as black
    
    return projection

@app.route('/frame', methods=['POST'])
def receive_rgb():
    try:
        # Convert received image to numpy array
        img = Image.open(io.BytesIO(request.data)).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid frame shape: {frame.shape}")

        # Store original frame
        state["latest_frame"] = frame

        # Get depth map
        depth_map, depth_error = depth_estimator.estimate_depth(frame)
        if depth_error:
            print(f"Depth estimation error: {depth_error}")
            return "Error", 500

        # Store depth map and visualization
        state["latest_depth"] = depth_map
        state["latest_depth_vis"] = depth_estimator.get_visualization(depth_map)
        
        # Create simple projection
        state["latest_projection"] = project_depth_to_ground(depth_map)

        return "OK", 200

    except Exception as e:
        print("Error in /frame:", e)
        traceback.print_exc()
        return "Error", 500

def display_loop():
    """Display the latest frames and depth visualization"""
    while True:
        try:
            # Display RGB frame
            if state["latest_frame"] is not None:
                cv2.imshow("RGB Frame", state["latest_frame"])
            
            # Display depth visualization
            if state["latest_depth_vis"] is not None:
                cv2.imshow("Depth Map", state["latest_depth_vis"])
                
            # Display projection
            if state["latest_projection"] is not None:
                cv2.imshow("Ground Projection", state["latest_projection"])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            traceback.print_exc()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("🟢 Flask server running at http://127.0.0.1:5000")
    print("🎥 RGB stream:         http://127.0.0.1:5000/frame")
    
    # Start display loop in a separate thread
    import threading
    threading.Thread(target=display_loop, daemon=True).start()
    
    # Start Flask server
    app.run(host="0.0.0.0", port=5000) 