#!/usr/bin/env python3
"""
Precision landing main loop - AprilTag detection wired into FCLink.

Detects an AprilTag fiducial via the downward-facing camera and streams
MAVLink LANDING_TARGET messages to the flight controller through the
FCLink class, for ArduPilot's Precision Landing/Loiter feature.

Headless by design (no cv2.imshow) - this is meant to run unattended on
the Pi during actual flight, not at a bench with a monitor attached.

Requires camera_calibration.yaml in this directory (see
camera_calibration_tool.py) and fc_link.py on the same path.
"""

import time
import cv2
import numpy as np
from pyapriltags import Detector

from fc_link import FCLink

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
TARGET_FPS = 60

CALIB_FILE = 'camera_calibration.yaml'
TAG_FAMILY = 'tag36h11'
TAG_ID = None            # None = accept any tag ID; set an int to filter
TAG_SIZE_M = 0.17         # MUST match your printed tag exactly

SEND_RATE_HZ = 20         # LANDING_TARGET send rate (10-50 Hz recommended)

# ---------------------------------------------------------------------------
# Load camera calibration
# ---------------------------------------------------------------------------
fs = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode('camera_matrix').mat()
fs.release()

if camera_matrix is None:
    raise RuntimeError(f"Could not load camera_matrix from {CALIB_FILE}")

FX, FY = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
CX, CY = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])

# ---------------------------------------------------------------------------
# AprilTag detector
# ---------------------------------------------------------------------------
detector = Detector(
    families=TAG_FAMILY,
    nthreads=4,
    quad_decimate=1.5,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
)

# ---------------------------------------------------------------------------
# Camera capture (USB/V4L2)
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

# ---------------------------------------------------------------------------
# Flight controller link
# ---------------------------------------------------------------------------
link = FCLink()

# If you also want to react to incoming telemetry (battery, attitude, etc.)
# while this detection loop runs, register handlers and run link.spin() in
# a background thread instead of calling it directly here - spin() blocks,
# and this script's own camera loop needs to be the main loop. Example:
#
#   import threading
#   threading.Thread(target=link.spin, daemon=True).start()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
min_send_interval = 1.0 / SEND_RATE_HZ
last_send_time = 0.0
consecutive_failures = 0
MAX_CONSECUTIVE_FAILURES = 30

print("Starting precision landing detection loop. Ctrl+C to stop.")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError("Camera repeatedly failed to deliver frames.")
            continue
        consecutive_failures = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(FX, FY, CX, CY),
            tag_size=TAG_SIZE_M,
        )

        candidates = [d for d in detections if (TAG_ID is None or d.tag_id == TAG_ID)]
        now = time.time()

        if candidates and (now - last_send_time) >= min_send_interval:
            # Prefer the largest (likely closest/most reliable) detection
            def tag_area(d):
                c = d.corners
                return 0.5 * abs(
                    (c[0][0] * c[1][1] - c[1][0] * c[0][1])
                    + (c[1][0] * c[2][1] - c[2][0] * c[1][1])
                    + (c[2][0] * c[3][1] - c[3][0] * c[2][1])
                    + (c[3][0] * c[0][1] - c[0][0] * c[3][1])
                )

            tag = max(candidates, key=tag_area)

            u, v = tag.center
            angle_x = float(np.arctan2(u - CX, FX))
            angle_y = float(np.arctan2(v - CY, FY))
            distance = float(tag.pose_t[2][0])

            corner_px = np.array(tag.corners, dtype=np.float64)
            corner_angles_x = np.arctan2(corner_px[:, 0] - CX, FX)
            corner_angles_y = np.arctan2(corner_px[:, 1] - CY, FY)
            size_x = float(corner_angles_x.max() - corner_angles_x.min())
            size_y = float(corner_angles_y.max() - corner_angles_y.min())

            link.send_landing_target(
                angle_x=angle_x,
                angle_y=angle_y,
                distance=distance,
                size_x=size_x,
                size_y=size_y,
                target_num=tag.tag_id,
            )
            last_send_time = now

except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    cap.release()