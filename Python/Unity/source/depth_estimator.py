import torch
import cv2
import numpy as np
from typing import Optional, Tuple

class DepthEstimator:
    def __init__(self, model_type: str = "MiDaS_small", device: Optional[str] = None):
        """
        Initialize the depth estimator.
        
        Args:
            model_type: One of 'DPT_Large', 'DPT_Hybrid', or 'MiDaS_small'
            device: Device to run the model on ('cuda' or 'cpu'). If None, will auto-detect.
        """
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        """Load the MiDaS model and its transforms."""
        print(f"📦 Loading MiDaS model ({self.model_type}) on {self.device}...")

        # Load model
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        # Load proper transform
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

    def estimate_depth(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Estimate depth from an RGB/BGR image.
        
        Args:
            frame: Input image in BGR format (OpenCV format)
            
        Returns:
            Tuple of (depth_map, error_message)
            - depth_map: Normalized depth map (0-1) or None if error
            - error_message: Error message if something went wrong, None otherwise
        """
        try:
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                return None, f"Invalid frame shape: {frame.shape if frame is not None else None}"

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Transform and move to device
            input_tensor = self.transform(img_rgb).to(self.device)

            # Get prediction
            with torch.no_grad():
                prediction = self.model(input_tensor)

            # Interpolate to original size (256x256)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(256, 256),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            # Convert to numpy and normalize
            depth_map = prediction.cpu().numpy().astype(np.float32)
            depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

            return depth_map, None

        except Exception as e:
            return None, f"Depth estimation error: {str(e)}"

    def get_visualization(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Create a colorized visualization of the depth map.
        
        Args:
            depth_map: Normalized depth map (0-1)
            
        Returns:
            Colorized depth map in BGR format
        """
        if depth_map is None:
            return None
            
        depth_vis = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA) 