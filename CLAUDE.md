\# UAV: AprilTag precision docking



Autonomous vision-based docking for a custom 5" quad. Companion computer

runs the CV pipeline, sends MAVLink LANDING\_TARGET to the flight controller.



\## Hardware

\- Companion: Raspberry Pi 3B+, 1GB RAM, aarch64, Debian Trixie, console-only boot

\- Camera: ELP AR0234 global shutter USB (UVC), /dev/video0, \~126 deg HFOV

\- FC: MicoAir743 v2, ArduCopter 4.7.0

\- Rangefinder: Benewake TF-Luna, downward, 0.2m minimum range



\## Running code on the Pi

All CV code must run on the Pi (camera is attached there). Reach it via:

&#x20; ssh UAV@raspberrypi.local "cd \~/UAV \&\& source .venv/bin/activate \&\& <cmd>"

Repo is cloned at \~/UAV on the Pi. Pull there after pushing from Windows.



\## Environment

\- venv at \~/UAV/.venv, created with --system-site-packages (required: OpenCV is apt-installed)

\- OpenCV 4.10.0 from apt (python3-opencv). Use the NEW ArUco API:

&#x20; cv2.aruco.getPredefinedDictionary + cv2.aruco.ArucoDetector.

&#x20; Do NOT use cv2.aruco.Dictionary\_get or DetectorParameters\_create (pre-4.7 API).

\- Camera is MJPG-only in practice. YUYV caps at 5fps @1920x1200. Always set

&#x20; CAP\_PROP\_FOURCC to MJPG \*before\* setting width/height.



\## Constraints

\- 1GB RAM total. Do not hold multiple full-res frames. Prefer streaming over batching.

\- Pi 3B+ CPU is the bottleneck, not USB bandwidth. MJPEG decode is the dominant cost.

\- Benchmarks must run over plain SSH with editors disconnected, or numbers are skewed.



\## Open questions (unresolved, do not assume)

\- Do sub-native resolutions scale or center-crop? Affects FOV and intrinsics. UNTESTED.

\- Actual MJPEG decode time at 1600x1200 on this Pi. UNMEASURED.

\- Whether OpenCV ArUco's detector has enough range vs. native AprilTag lib. UNTESTED.



\## Style

Python 3. Never write per-pixel loops in Python; use numpy or OpenCV calls.

