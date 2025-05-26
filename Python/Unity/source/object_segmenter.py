from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

# Disable YOLO logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

@dataclass
class Detection:
    """Container for a single detection."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    label: str
    confidence: float
    mask: Optional[np.ndarray] = None

class ObjectSegmenter:
    def __init__(self, model_path: str = "yolov8s-seg.pt", conf_threshold: float = 0.4):
        """
        Initialize the object segmenter.
        
        Args:
            model_path: Path to the YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.color_map: Dict[str, Tuple[int, int, int]] = {}

    def detect_and_segment(self, frame: np.ndarray) -> Tuple[List[Detection], Optional[str]]:
        """
        Detect and segment objects in an image.
        
        Args:
            frame: Input image in BGR format
            
        Returns:
            Tuple of (detections, error_message)
            - detections: List of Detection objects
            - error_message: Error message if something went wrong, None otherwise
        """
        try:
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                return [], f"Invalid frame shape: {frame.shape if frame is not None else None}"

            # Run inference with 256x256 input size
            device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
            results = self.model.predict(frame, imgsz=256, conf=self.conf_threshold, device=device, verbose=False)[0]

            detections = []
            
            # Process boxes
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                cls_ids = results.boxes.cls.cpu().numpy()
                
                # Process masks if available
                masks = None
                if results.masks is not None:
                    masks = results.masks.data.cpu().numpy()

                # Create Detection objects
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
                    label = self.model.names[int(cls_id)]
                    mask = masks[i] if masks is not None else None
                    
                    detection = Detection(
                        bbox=tuple(map(int, box)),
                        label=label,
                        confidence=float(conf),
                        mask=mask
                    )
                    detections.append(detection)

            return detections, None

        except Exception as e:
            return [], f"Detection error: {str(e)}"

    def get_visualization(self, frame: np.ndarray, detections: List[Detection]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create visualization of detections and segmentations.
        
        Args:
            frame: Original image in BGR format
            detections: List of Detection objects
            
        Returns:
            Tuple of (detection_vis, segmentation_vis)
            - detection_vis: Image with bounding boxes and labels
            - segmentation_vis: Image with segmentation masks and labels
        """
        detection_vis = frame.copy()
        segmentation_vis = frame.copy()

        for det in detections:
            # Get color for this class
            color = self._get_color_for_class(det.label)
            
            # Draw bounding box
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(detection_vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(detection_vis, det.label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw segmentation mask if available
            if det.mask is not None:
                mask_binary = (det.mask > 0.3).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    if cv2.contourArea(cnt) > 100:  # Ignore tiny segments
                        cv2.drawContours(segmentation_vis, [cnt], -1, color, thickness=2)

                # Label centroid
                M = cv2.moments(mask_binary)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(segmentation_vis, det.label, (cX, cY), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return detection_vis, segmentation_vis

    def _get_color_for_class(self, label: str) -> Tuple[int, int, int]:
        """Get a consistent color for a class label."""
        if label not in self.color_map:
            self.color_map[label] = tuple(np.random.randint(0, 255, size=3).tolist())
        return self.color_map[label] 