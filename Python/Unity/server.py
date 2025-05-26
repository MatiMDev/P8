from flask import Flask, request
import logging
import io
import traceback
import numpy as np
import cv2
from PIL import Image

from config import DISABLE_FLASK_LOGGING, DISABLE_ULTRALYTICS_LOGGING, MODEL_TYPE
from utils.streamer import Streamer
from video_processor.depth_estimator import MiDaSDepthEstimator
from video_processor.object_segmenter import ObjectSegmenter
from data_processor import (
    process_rgb_frame,
    process_bev_frame,
    process_groundtruth_data,
    process_imu_data,
)

# ──────────────────────────────
# LOGGING
if DISABLE_FLASK_LOGGING:
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
if DISABLE_ULTRALYTICS_LOGGING:
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ──────────────────────────────
# APP & STREAMER
app = Flask(__name__)
streamer = Streamer()
app.register_blueprint(streamer.blueprint)

# ──────────────────────────────
# COMPONENTS
depth_estimator = MiDaSDepthEstimator(model_type=MODEL_TYPE)
segmenter = ObjectSegmenter()

# ──────────────────────────────
# SHARED STATE
state = {
    "latest_frame": None,
    "latest_depth": None,
    "latest_bev_frame": None,
    "latest_detection": None,
    "latest_segmentation": None,
    "groundtruth": None,
    "imu": None,
}

# ──────────────────────────────
# ROUTES

@app.route('/frame', methods=['POST'])
def receive_rgb():
    try:
        img = Image.open(io.BytesIO(request.data)).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid frame shape: {frame.shape}")

        process_rgb_frame(frame, depth_estimator, streamer, state)
        return "OK", 200

    except Exception as e:
        print("Error in /frame:", e)
        traceback.print_exc()
        return "Error", 500


@app.route('/frame-bev', methods=['POST'])
def receive_bev():
    try:
        img = Image.open(io.BytesIO(request.data)).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid BEV frame shape: {frame.shape}")

        process_bev_frame(frame, streamer, state)
        return "OK", 200

    except Exception as e:
        print("Error in /frame-bev:", e)
        traceback.print_exc()
        return "Error", 500


@app.route('/groundtruth', methods=['POST'])
def receive_groundtruth():
    try:
        data = request.get_json()
        process_groundtruth_data(data, state)

        # print("vehicle", {
        #     "timestamp": {
        #         "groundtruth": data["timestamp"],
        #         "imu": state["imu"]["timestamp"] if state["imu"] else None
        #     },
        #     "groundtruth": state.get("groundtruth", {}),
        #     "imu": state.get("imu", {})
        # })


        streamer.update_data("vehicle", {
            "timestamp": {
                "groundtruth": data["timestamp"],
                "imu": state["imu"]["timestamp"] if state["imu"] else None
            },
            "groundtruth": state.get("groundtruth", {}),
            "imu": state.get("imu", {})
        })

        return "OK", 200

    except Exception as e:
        print("Error in /groundtruth:", e)
        traceback.print_exc()
        return "Error", 500


@app.route('/imu', methods=['POST'])
def receive_imu():
    try:
        data = request.get_json()
        process_imu_data(data, state)


        streamer.update_data("vehicle", {
            "timestamp": {
                "groundtruth": state["groundtruth"]["timestamp"] if state["groundtruth"] else None,
                "imu": data["timestamp"]
            },
            "groundtruth": state.get("groundtruth", {}),
            "imu": state.get("imu", {})
        })

        return "OK", 200

    except Exception as e:
        print("Error in /imu:", e)
        traceback.print_exc()
        return "Error", 500

# ──────────────────────────────
# EXPORTS
__all__ = ['app', 'state', 'streamer']
