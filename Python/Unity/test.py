from flask import Flask, request
from PIL import Image
import io
import numpy as np
import cv2
import threading
import traceback
import logging

from utils.video_streamer import VideoStreamer
from video_processor.depth_estimator import MiDaSDepthEstimator
from video_processor.object_segmenter import ObjectSegmenter

# ──────────────────────────────────────────────────────────────
# ⚙️ CONFIGURATION
SHOW_WINDOWS = True
DISABLE_FLASK_LOGGING = True
DISABLE_ULTRALYTICS_LOGGING = True

if DISABLE_FLASK_LOGGING:
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
if DISABLE_ULTRALYTICS_LOGGING:
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ──────────────────────────────────────────────────────────────
# 🔧 APP SETUP
app = Flask(__name__)
latest_frame = None
latest_depth = None
latest_bev_frame = None
latest_detection = None
latest_segmentation = None

streamer = VideoStreamer()
app.register_blueprint(streamer.blueprint)

depth_estimator = MiDaSDepthEstimator(model_type="MiDaS_small")
segmenter = ObjectSegmenter()

# ──────────────────────────────────────────────────────────────
# 👁️ DISPLAY LOOP
def display_loop():
    while True:
        try:
            if latest_frame is not None:
                cv2.imshow("Unity Camera Feed", latest_frame)

            if latest_bev_frame is not None:
                cv2.imshow("Top-Down BEV View", latest_bev_frame)

            if latest_depth is not None and latest_depth.ndim == 2:
                depth_vis = (latest_depth * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
                cv2.imshow("Depth Estimation", depth_colored)

            if latest_detection is not None:
                cv2.imshow("Object Detection", latest_detection)

            if latest_segmentation is not None:
                cv2.imshow("Object Segmentation", latest_segmentation)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print("Display error:", e)
            traceback.print_exc()
    cv2.destroyAllWindows()

# ──────────────────────────────────────────────────────────────
# 📥 FRONT CAMERA FRAME
@app.route('/frame', methods=['POST'])
def receive_frame():
    global latest_frame, latest_depth
    try:
        img = Image.open(io.BytesIO(request.data)).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid frame shape: {frame.shape}")

        latest_frame = frame
        latest_depth = depth_estimator.estimate_depth(frame)
        streamer.update_frame("rgb", frame)

        if latest_depth is not None:
            depth_vis = (latest_depth * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
            streamer.update_frame("depth", depth_colored)

        return "OK", 200

    except Exception as e:
        print("Error receiving frame:", e)
        traceback.print_exc()
        return "Error", 500

# ──────────────────────────────────────────────────────────────
# 📥 BEV CAMERA FRAME
@app.route('/frame-bev', methods=['POST'])
def receive_bev_frame():
    global latest_bev_frame
    try:
        img = Image.open(io.BytesIO(request.data)).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid BEV frame shape: {frame.shape}")

        latest_bev_frame = frame
        streamer.update_frame("bev", frame)

        return "OK", 200

    except Exception as e:
        print("Error receiving BEV frame:", e)
        traceback.print_exc()
        return "Error", 500

# ──────────────────────────────────────────────────────────────
# 🚀 RUN SERVER
if __name__ == '__main__':
    print("🟢 Flask server running at http://127.0.0.1:5000")
    print("🎥 RGB stream:         http://127.0.0.1:5000/video/rgb")
    print("🌊 Depth stream:       http://127.0.0.1:5000/video/depth")
    print("⬛ BEV stream:         http://127.0.0.1:5000/video/bev")

    if SHOW_WINDOWS:
        threading.Thread(target=display_loop, daemon=True).start()

    app.run(host="0.0.0.0", port=5000)