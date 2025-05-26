import cv2
import numpy as np
from object_segmenter import ObjectSegmenter

def main():
    # Initialize the object segmenter
    segmenter = ObjectSegmenter(model_path="yolov8s-seg.pt", conf_threshold=0.4)
    
    # Initialize video capture (0 for webcam, or provide video file path)
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Detect and segment objects
        detections, error = segmenter.detect_and_segment(frame)
        
        if error:
            print(f"Error: {error}")
            continue
            
        # Get visualizations
        detection_vis, segmentation_vis = segmenter.get_visualization(frame, detections)
        
        # Display results
        cv2.imshow("Object Detection", detection_vis)
        cv2.imshow("Instance Segmentation", segmentation_vis)
        
        # Print detections
        for det in detections:
            print(f"Detected {det.label} with confidence {det.confidence:.2f}")
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 