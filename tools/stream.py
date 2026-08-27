#!/usr/bin/env python3
"""MJPEG HTTP streamer for /dev/video0 (MJPG UVC).

Serves multipart/x-mixed-replace so a browser or VLC on the ground station can
watch the camera live. One capture thread owns the device and publishes only the
most recent JPEG; handler threads send whatever is current, so slow clients drop
frames instead of queueing full-res buffers in RAM.

By default the JPEG the camera produced is forwarded untouched (no decode, no
re-encode) -- see camera_open() for how that mode is negotiated.
"""

import argparse
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

BOUNDARY = "uavframe"

INDEX_PAGE = b"""<!doctype html>
<title>UAV camera</title>
<style>body{margin:0;background:#111}img{width:100%;height:auto;display:block}</style>
<img src="/stream.mjpg">
"""


def negotiated_fourcc(cap):
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))


def trim_jpeg(buf):
    """Return the JPEG in a raw V4L2 buffer, or None if it does not hold one.

    The driver hands back a fixed-size buffer that may be padded past the EOI
    marker, so the tail has to be cut off before the bytes are a valid image.
    """
    if len(buf) < 4 or not buf.startswith(b"\xff\xd8"):
        return None
    end = buf.rfind(b"\xff\xd9")
    if end < 0:
        return None
    end += 2
    return buf if end == len(buf) else buf[:end]


class FrameBroker:
    """Single-slot latest-frame handoff from the capture thread to clients."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0
        self._stopped = False

    def publish(self, jpeg):
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    @property
    def stopped(self):
        return self._stopped

    def wait_for(self, last_seq, timeout=5.0):
        """Block until a frame newer than last_seq exists. Returns (seq, jpeg)."""
        with self._cond:
            if self._seq <= last_seq and not self._stopped:
                self._cond.wait(timeout)
            if self._stopped or self._seq <= last_seq:
                return None
            return self._seq, self._jpeg


def camera_open(args):
    """Open the camera and decide whether raw JPEG pass-through is available.

    Returns (cap, passthrough). With CAP_PROP_CONVERT_RGB cleared the V4L2
    backend skips its MJPEG decode and retrieve() yields the compressed
    bitstream as a flat uint8 array -- the whole point of this tool on a Pi 3B+,
    where the decode is the dominant per-frame cost. Not every OpenCV build
    honours it, so the result is probed on a real frame rather than trusted.
    """
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: could not open {args.device}", file=sys.stderr)
        sys.exit(1)

    # MJPG must be negotiated before width/height, or the driver falls back to YUYV.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    passthrough = False
    if not args.decode:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None and frame.dtype == np.uint8:
                if trim_jpeg(frame.reshape(-1).tobytes()) is not None:
                    passthrough = True
                    break
        if not passthrough:
            # Backend ignored the request; reopen so the decode path starts clean.
            cap.release()
            args.decode = True
            return camera_open(args)

    print(
        f"negotiated: fourcc={negotiated_fourcc(cap)} "
        f"resolution={int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"fps={cap.get(cv2.CAP_PROP_FPS):.1f}"
    )
    return cap, passthrough


def capture_loop(cap, passthrough, broker, args, stats):
    """Own the camera; publish one JPEG per frame until stopped."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
    failures = 0

    while not broker.stopped:
        ok, frame = cap.read()
        if not ok or frame is None:
            failures += 1
            if failures >= 30:
                print("ERROR: camera stopped delivering frames", file=sys.stderr)
                break
            continue
        failures = 0

        if passthrough:
            raw = trim_jpeg(frame.reshape(-1).tobytes())
            if raw is None:
                continue
            if args.gray:
                # JPEG luma cannot be lifted out without decoding, so gray costs
                # a decode plus a re-encode even on the pass-through path.
                gray = cv2.imdecode(frame.reshape(-1), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                ok, enc = cv2.imencode(".jpg", gray, encode_params)
                if not ok:
                    continue
                jpeg = enc.tobytes()
            else:
                jpeg = raw
        else:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if args.gray else frame
            ok, enc = cv2.imencode(".jpg", image, encode_params)
            if not ok:
                continue
            jpeg = enc.tobytes()

        broker.publish(jpeg)
        stats.count(len(jpeg))

    cap.release()
    broker.stop()


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = 0
        self.bytes = 0
        self.clients = 0

    def count(self, nbytes):
        with self.lock:
            self.frames += 1
            self.bytes += nbytes

    def take(self):
        with self.lock:
            frames, nbytes = self.frames, self.bytes
            self.frames = self.bytes = 0
            return frames, nbytes, self.clients


def make_handler(broker, stats):
    class MJPEGHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "UAVStream/1.0"

        def log_message(self, fmt, *fmt_args):
            pass  # per-request logging would drown the periodic stats line

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send_bytes(INDEX_PAGE, "text/html")
            elif path == "/snapshot.jpg":
                self._send_snapshot()
            elif path in ("/stream.mjpg", "/stream"):
                self._send_stream()
            else:
                self.send_error(404)

        def _send_bytes(self, payload, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_snapshot(self):
            latest = broker.wait_for(0)
            if latest is None:
                self.send_error(503, "No frame available")
                return
            self._send_bytes(latest[1], "image/jpeg")

        def _send_stream(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.end_headers()
            self.close_connection = True

            with stats.lock:
                stats.clients += 1
            last_seq = 0
            try:
                while not broker.stopped:
                    latest = broker.wait_for(last_seq)
                    if latest is None:
                        continue  # capture stalled; headers already sent, keep waiting
                    last_seq, jpeg = latest
                    header = (
                        f"--{BOUNDARY}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n"
                    ).encode("ascii")
                    # One write per frame: fewer syscalls, and only one copy of
                    # the frame is alive at a time.
                    self.wfile.write(header + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout):
                pass  # client navigated away
            finally:
                with stats.lock:
                    stats.clients -= 1

    return MJPEGHandler


def main():
    parser = argparse.ArgumentParser(description="Serve /dev/video0 as MJPEG over HTTP")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--gray", action="store_true", help="stream grayscale (forces decode + re-encode)")
    parser.add_argument("--decode", action="store_true", help="force the decode/re-encode path")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality when re-encoding")
    parser.add_argument("--quiet", action="store_true", help="suppress the periodic stats line")
    args = parser.parse_args()

    cap, passthrough = camera_open(args)
    mode = "passthrough (no decode)" if passthrough and not args.gray else "decode + re-encode"
    print(f"mode: {mode}{' [gray]' if args.gray else ''}")

    broker = FrameBroker()
    stats = Stats()
    grabber = threading.Thread(
        target=capture_loop, args=(cap, passthrough, broker, args, stats), daemon=True
    )
    grabber.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(broker, stats))
    server.daemon_threads = True
    print(f"serving http://{args.host}:{args.port}/  (stream at /stream.mjpg, still at /snapshot.jpg)")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        last = time.monotonic()
        while not broker.stopped:
            time.sleep(5.0)
            frames, nbytes, clients = stats.take()
            now = time.monotonic()
            elapsed, last = now - last, now
            if not args.quiet and elapsed > 0:
                kb = (nbytes / frames / 1024) if frames else 0.0
                print(f"{frames / elapsed:5.1f} fps  {kb:6.1f} KB/frame  clients={clients}")
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        broker.stop()
        server.shutdown()
        server.server_close()
        grabber.join(timeout=2.0)


if __name__ == "__main__":
    main()
