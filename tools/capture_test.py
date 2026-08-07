#!/usr/bin/env python3
"""Camera capture benchmark for /dev/video0 (MJPG UVC)."""

import argparse
import statistics
import sys

import cv2


def negotiated_fourcc(cap):
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))


def main():
    parser = argparse.ArgumentParser(description="Benchmark grab/retrieve/decode timing on /dev/video0")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=100, help="frames to time after warmup")
    parser.add_argument("--save", default=None, help="path to save one decoded frame as PNG")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: could not open {args.device}", file=sys.stderr)
        sys.exit(1)

    # MJPG must be negotiated before width/height, or the driver falls back to YUYV.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"negotiated: fourcc={negotiated_fourcc(cap)} resolution={actual_w}x{actual_h} fps={actual_fps:.1f}")

    for _ in range(10):
        cap.grab()
        cap.retrieve()

    grab_times = []
    retrieve_times = []
    gray_times = []
    last_frame = None

    t_start = cv2.getTickCount()
    for i in range(args.frames):
        t0 = cv2.getTickCount()
        ok = cap.grab()
        t1 = cv2.getTickCount()
        if not ok:
            print(f"WARNING: grab failed on frame {i}", file=sys.stderr)
            continue

        ok, frame = cap.retrieve()
        t2 = cv2.getTickCount()
        if not ok:
            print(f"WARNING: retrieve failed on frame {i}", file=sys.stderr)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t3 = cv2.getTickCount()

        freq = cv2.getTickFrequency()
        grab_times.append((t1 - t0) / freq)
        retrieve_times.append((t2 - t1) / freq)
        gray_times.append((t3 - t2) / freq)
        last_frame = frame
    t_end = cv2.getTickCount()

    cap.release()

    n = len(grab_times)
    if n == 0:
        print("ERROR: no frames captured", file=sys.stderr)
        sys.exit(1)

    elapsed = (t_end - t_start) / cv2.getTickFrequency()
    effective_fps = n / elapsed if elapsed > 0 else 0.0

    print(f"frames captured: {n}/{args.frames}")
    print(f"median grab:     {statistics.median(grab_times) * 1000:.2f} ms")
    print(f"median retrieve: {statistics.median(retrieve_times) * 1000:.2f} ms")
    print(f"median BGR2GRAY: {statistics.median(gray_times) * 1000:.2f} ms")
    print(f"effective fps:   {effective_fps:.2f}")

    if args.save and last_frame is not None:
        cv2.imwrite(args.save, last_frame)
        print(f"saved frame to {args.save}")


if __name__ == "__main__":
    main()
