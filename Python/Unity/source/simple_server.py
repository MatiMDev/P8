from flask import Flask, request
import logging
import io
import traceback
import numpy as np
import cv2
from PIL import Image
from object_segmenter import ObjectSegmenter
from simple_mapper import simple_mapper

# Disable Flask logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Initialize Flask app
app = Flask(__name__)

# Initialize object segmenter
segmenter = ObjectSegmenter(model_path="yolov8s-seg.pt", conf_threshold=0.4)

# Global state
state = {
    "latest_frame": None,
    "latest_detections": None,
    "latest_detection_vis": None,
    "latest_segmentation_vis": None,
    "latest_map_vis": None
}

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

        # Detect and segment objects
        detections, error = segmenter.detect_and_segment(frame)
        if error:
            print(f"Detection error: {error}")
            return "Error", 500

        # Store detections
        state["latest_detections"] = detections

        # Get visualizations
        detection_vis, segmentation_vis = segmenter.get_visualization(frame, detections)
        state["latest_detection_vis"] = detection_vis
        state["latest_segmentation_vis"] = segmentation_vis

        # Project detections onto ground plane
        simple_mapper.project_detections(detections, frame.shape)
        state["latest_map_vis"] = simple_mapper.get_visualization()

        return "OK", 200

    except Exception as e:
        print("Error in /frame:", e)
        traceback.print_exc()
        return "Error", 500

@app.route('/detections', methods=['GET'])
def get_detections():
    """Return the latest detections as JSON"""
    if state["latest_detections"] is None:
        return {"error": "No detections available"}, 404

    detections = []
    for det in state["latest_detections"]:
        detections.append({
            "label": det.label,
            "confidence": float(det.confidence),
            "bbox": det.bbox
        })
    
    return {"detections": detections}

def display_loop():
    """Display the latest frames and detections"""
    while True:
        try:
            # Display RGB frame with detections
            if state["latest_detection_vis"] is not None:
                cv2.imshow("Object Detection", state["latest_detection_vis"])
            
            # Display segmentation visualization
            if state["latest_segmentation_vis"] is not None:
                cv2.imshow("Instance Segmentation", state["latest_segmentation_vis"])

            # Display top-down view
            if state["latest_map_vis"] is not None:
                cv2.imshow("Top-Down View", state["latest_map_vis"])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            traceback.print_exc()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("🟢 Flask server running at http://127.0.0.1:5000")
    print("🎥 RGB stream:         http://127.0.0.1:5000/frame")
    print("📦 Object detection:    http://127.0.0.1:5000/detections")
    
    # Start display loop in a separate thread
    import threading
    threading.Thread(target=display_loop, daemon=True).start()
    
    # Start Flask server
    app.run(host="0.0.0.0", port=5000) 