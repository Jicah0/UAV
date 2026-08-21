import time
import cv2
import numpy as np
from pyapriltags import Detector

# ---------------------------------------------------------------------------
# CAMERA CALIBRATION - REPLACE THESE WITH YOUR ACTUAL CALIBRATION VALUES
# ---------------------------------------------------------------------------
# angle_x, angle_y, size_x, size_y, distance, x, y, z and q ALL depend on
# these being correct for YOUR specific camera + lens. Run a standard
# OpenCV chessboard/charuco calibration and replace the placeholders below
# (or load them from a saved camera_calibration.yaml - see commented block).
FX, FY = 500.0, 500.0   # focal length in pixels
CX, CY = 320.0, 240.0   # principal point in pixels (usually near frame center)

# fs = cv2.FileStorage('camera_calibration.yaml', cv2.FILE_STORAGE_READ)
# camera_matrix = fs.getNode('camera_matrix').mat()
# fs.release()
# FX, FY = camera_matrix[0, 0], camera_matrix[1, 1]
# CX, CY = camera_matrix[0, 2], camera_matrix[1, 2]

TAG_SIZE_M = 0.17  # physical black-square side length of your printed tag, meters

# LANDING_TARGET_TYPE enum (mavlink common.xml) - AprilTag is a fiducial marker
LANDING_TARGET_TYPE_VISION_FIDUCIAL = 2

# Initialize the detector for the standard FRC/Robotics family
# Common families include: 'tag36h11', 'tag25h9', 'tag16h5'
detector = Detector(families="tag36h11")


def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return (qw, qx, qy, qz)


# Capture video from default webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for AprilTag processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Execute detection loop - estimate_tag_pose=True + camera_params + tag_size
    # are required to get pose_R / pose_t populated on each detection
    results = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(FX, FY, CX, CY),
        tag_size=TAG_SIZE_M,
    )

    time_usec = int(time.time() * 1e6)  # LANDING_TARGET.time_usec

    # Process all detected tags in the current frame
    for r in results:
        # Extract the corner pixel locations (4 coordinate pairs)
        (ptA, ptB, ptC, ptD) = r.corners
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))

        # Draw the bounding box perimeter
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)

        # Mark the center coordinates
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        # Overlay the unique Tag ID integer
        cv2.putText(frame, f"ID: {r.tag_id}", (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # -------------------------------------------------------------
        # Compute all fields needed for the MAVLink LANDING_TARGET message
        # -------------------------------------------------------------

        # target_num: ID of the target if multiple targets are present
        target_num = r.tag_id

        # frame: coordinate frame for the position/quaternion fields.
        # ArduPilot only supports the position/quaternion form in
        # MAV_FRAME_BODY_FRD. NOTE: pose_t/pose_R below are in the
        # CAMERA's own optical frame (OpenCV convention: x-right,
        # y-down, z-forward out of the lens) - NOT automatically the
        # vehicle body FRD frame. If your camera isn't mounted perfectly
        # aligned with vehicle-forward, you must rotate x/y/z and q into
        # body FRD before sending, or ArduPilot will steer the wrong way.
        mav_frame = "MAV_FRAME_BODY_FRD"

        # angle_x / angle_y: angular offset of tag center from image
        # center, in radians (image-relative form)
        px_off_x = r.center[0] - CX
        px_off_y = r.center[1] - CY
        angle_x = float(np.arctan2(px_off_x, FX))
        angle_y = float(np.arctan2(px_off_y, FY))

        # distance: straight-line distance to target, meters
        # (z-component of the camera-frame translation vector)
        distance = float(r.pose_t[2][0])

        # size_x / size_y: angle spanned by the tag's extent in each axis
        # ("angle between the smallest and biggest pixel in x/y direction")
        corner_px = np.array([ptA, ptB, ptC, ptD], dtype=np.float64)
        corner_angles_x = np.arctan2(corner_px[:, 0] - CX, FX)
        corner_angles_y = np.arctan2(corner_px[:, 1] - CY, FY)
        size_x = float(corner_angles_x.max() - corner_angles_x.min())
        size_y = float(corner_angles_y.max() - corner_angles_y.min())

        # x / y / z: position of the target in the specified MAV_FRAME, meters
        # (raw camera-optical-frame translation - see frame caveat above)
        x, y, z = (float(r.pose_t[0][0]), float(r.pose_t[1][0]), float(r.pose_t[2][0]))

        # q: orientation quaternion (w, x, y, z order), also camera-frame
        q = rotation_matrix_to_quaternion(r.pose_R)

        # type: type of landing target
        target_type = LANDING_TARGET_TYPE_VISION_FIDUCIAL

        # position_valid: whether x/y/z/q are usable. True here since
        # estimate_tag_pose succeeded for this detection (pose_err gives
        # you a quality signal if you want to threshold this yourself,
        # e.g. position_valid = 1 if r.pose_err < some_threshold else 0)
        position_valid = 1

        # -------------------------------------------------------------
        # Display everything
        # -------------------------------------------------------------
        print(
            f"--- Tag {r.tag_id} @ t={time_usec} ---\n"
            f"  target_num      = {target_num}\n"
            f"  frame           = {mav_frame}\n"
            f"  angle_x (rad)   = {angle_x:+.4f}  ({np.degrees(angle_x):+.2f} deg)\n"
            f"  angle_y (rad)   = {angle_y:+.4f}  ({np.degrees(angle_y):+.2f} deg)\n"
            f"  distance (m)    = {distance:.3f}\n"
            f"  size_x (rad)    = {size_x:.4f}\n"
            f"  size_y (rad)    = {size_y:.4f}\n"
            f"  x, y, z (m)     = {x:+.3f}, {y:+.3f}, {z:+.3f}\n"
            f"  q (w,x,y,z)     = {q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f}\n"
            f"  type            = {target_type} (VISION_FIDUCIAL)\n"
            f"  position_valid  = {position_valid}\n"
            f"  pose_err (diag) = {r.pose_err:.2e}   [not a LANDING_TARGET field - detection quality only]"
        )

        # On-frame overlay of the key fields, stacked below the tag
        overlay_lines = [
            f"ang_x:{np.degrees(angle_x):+.1f}d ang_y:{np.degrees(angle_y):+.1f}d",
            f"dist:{distance:.2f}m size:{np.degrees(size_x):.1f}/{np.degrees(size_y):.1f}d",
            f"xyz:{x:+.2f},{y:+.2f},{z:+.2f}",
            f"q:{q[0]:.2f},{q[1]:.2f},{q[2]:.2f},{q[3]:.2f}",
        ]
        for i, line in enumerate(overlay_lines):
            cv2.putText(frame, line, (ptA[0], ptA[1] + 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    # Render frame
    cv2.imshow("AprilTag Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()