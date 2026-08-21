#!/usr/bin/env python3
"""
Camera Intrinsic Calibration Tool
==================================

THIS PROGRAM REQUIRES A MONITOR TO PREVIEW YOUR SAMPLE IMAGES

Two-phase workflow for measuring your webcam/global-shutter camera's
intrinsics (fx, fy, cx, cy, distortion coefficients), output in a
camera_calibration.yaml compatible with the earlier precision-landing
scripts (they load camera_matrix / dist_coeffs from this same file).

Usage:
    python camera_calibration_tool.py capture
        Live preview: shows a green overlay whenever a full chessboard is
        detected. Press 'c' to save that frame, 'q' to finish.
        Aim for 20-30+ images covering different angles, distances, and
        especially the EDGES/CORNERS of the frame (not just the center).

    python camera_calibration_tool.py calibrate
        Runs after 'capture'. Processes every saved image, solves for
        the camera matrix + distortion coefficients, prints the
        reprojection error, and writes camera_calibration.yaml.
        Also shows a before/after undistort comparison on one image
        as a visual sanity check.

Before you start:
    Print a chessboard pattern (e.g. OpenCV's standard 9x6-internal-corner
    board: https://github.com/opencv/opencv/blob/4.x/doc/pattern.png)
    on a rigid, flat surface. Measure the actual printed square size with
    calipers and set SQUARE_SIZE_M below.
"""

import sys
import os
import glob
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURATION - edit to match your printed board and camera
# ---------------------------------------------------------------------------
CHECKERBOARD = (9, 6)          # INTERNAL corners (not squares!) - width, height
SQUARE_SIZE_M = 0.025          # measured physical size of one square, meters
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
IMAGE_DIR = "calib_images"
OUTPUT_YAML = "camera_calibration.yaml"
MIN_RECOMMENDED_IMAGES = 20

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
FIND_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_FAST_CHECK
    + cv2.CALIB_CB_NORMALIZE_IMAGE
)


def capture_mode():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    existing = len(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    count = existing

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    # Report what the driver actually negotiated - it may silently differ
    # from FRAME_W/FRAME_H if that exact mode isn't supported.
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Requested {FRAME_W}x{FRAME_H}, camera reports {actual_w}x{actual_h}")

    print(f"Starting capture. {existing} image(s) already saved in {IMAGE_DIR}/")
    print("Move the board to a new angle/position, wait for green corners, "
          "press 'c' to save. Press 'q' when done (aim for "
          f"{MIN_RECOMMENDED_IMAGES}+ images).")

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 30  # fail loudly instead of spinning forever

    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            print(f"Frame read failed ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                cap.release()
                cv2.destroyAllWindows()
                raise RuntimeError(
                    "Camera opened but repeatedly failed to deliver frames. "
                    "Try removing the cap.set(FRAME_W/FRAME_H) calls above, "
                    "or check `v4l2-ctl --list-formats-ext -d /dev/video"
                    f"{CAMERA_INDEX}` for modes this camera actually supports."
                )
            continue
        consecutive_failures = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, FIND_FLAGS)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)

        cv2.putText(display, f"Saved: {count}  (target {MIN_RECOMMENDED_IMAGES}+)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if found else (0, 0, 255), 2)

        cv2.imshow("Calibration Capture - 'c' to save, 'q' to quit", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and found:
            path = os.path.join(IMAGE_DIR, f"img_{count:03d}.png")
            cv2.imwrite(path, frame)
            count += 1
            print(f"Saved {path}  (total: {count})")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {count} images saved in {IMAGE_DIR}/.")
    if count < MIN_RECOMMENDED_IMAGES:
        print(f"WARNING: fewer than {MIN_RECOMMENDED_IMAGES} images - "
              "calibration quality may suffer. Consider capturing more.")
    print("Next: python camera_calibration_tool.py calibrate")


def calibrate_mode():
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    if len(images) < 5:
        raise RuntimeError(
            f"Only found {len(images)} images in {IMAGE_DIR}/. "
            "Run 'capture' mode first and collect more images."
        )

    # 3D object points for one chessboard, in the board's own coordinate
    # frame (z=0 plane), scaled by the real physical square size
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    objpoints = []   # 3D points, one set per accepted image
    imgpoints = []   # corresponding 2D points, one set per accepted image
    image_size = None
    accepted = 0

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]   # (width, height)
        elif gray.shape[::-1] != image_size:
            print(f"SKIPPING {fname}: resolution {gray.shape[::-1]} doesn't "
                  f"match the rest of the set ({image_size}). "
                  "Mixed resolutions will corrupt the calibration.")
            continue

        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, FIND_FLAGS)
        if not found:
            print(f"SKIPPING {fname}: chessboard not detected.")
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        accepted += 1

    print(f"\n{accepted}/{len(images)} images used for calibration.")
    if accepted < 5:
        raise RuntimeError("Too few valid images to calibrate reliably.")

    reproj_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    print(f"\nReprojection error (RMS, pixels): {reproj_error:.4f}")
    if reproj_error < 0.5:
        print("  -> Good calibration.")
    elif reproj_error < 1.0:
        print("  -> Usable, but consider capturing more/better images.")
    else:
        print("  -> Poor calibration - recapture with more angle/edge "
              "coverage, better lighting, and a rigid (non-warped) board.")

    print(f"\ncamera_matrix (fx, fy, cx, cy) = {fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f}")
    print(f"dist_coeffs (k1, k2, p1, p2, k3) = {dist_coeffs.ravel()}")

    fs = cv2.FileStorage(OUTPUT_YAML, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("reprojection_error", reproj_error)
    fs.release()
    print(f"\nSaved {OUTPUT_YAML} - drop this next to your LANDING_TARGET "
          "script; it already loads camera_matrix/dist_coeffs from this file.")

    # Visual sanity check: undistort the first accepted image and show
    # it side-by-side with the original
    sample = cv2.imread(images[0])
    h, w = sample.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(sample, camera_matrix, dist_coeffs, None, new_mtx)
    comparison = np.hstack((sample, undistorted))
    cv2.imshow("Original (left) vs Undistorted (right) - press any key to close", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("capture", "calibrate"):
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "capture":
        capture_mode()
    else:
        calibrate_mode()