from flask import Flask, request
import logging
import io
import traceback
import numpy as np
import cv2
from PIL import Image

from config import DISABLE_FLASK_LOGGING
from data_processor import data_processor

# ──────────────────────────────
# LOGGING
if DISABLE_FLASK_LOGGING:
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

# ──────────────────────────────
# APP SETUP
app = Flask(__name__)

# ──────────────────────────────
# ROUTES
@app.route('/frame', methods=['POST'])
def receive_rgb():
    try:
        img = Image.open(io.BytesIO(request.data)).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid frame shape: {frame.shape}")

        data_processor.process_rgb_frame(frame)
        return "OK", 200

    except Exception as e:
        print("Error in /frame:", e)
        traceback.print_exc()
        return "Error", 500

@app.route('/groundtruth', methods=['POST'])
def receive_groundtruth():
    try:
        data = request.get_json()
        #print(data)
        data_processor.process_groundtruth_data(data)
        return "OK", 200

    except Exception as e:
        print("Error in /groundtruth:", e)
        traceback.print_exc()
        return "Error", 500

# ──────────────────────────────
# EXPORTS
__all__ = ['app'] 