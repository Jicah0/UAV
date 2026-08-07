#!/usr/bin/env python3
"""
Modular pymavlink companion-computer link: Raspberry Pi <-> ArduPilot FC
over UART.

Wiring:
    Pi GPIO14 (pin 8, TXD)  -> FC UARTx RX
    Pi GPIO15 (pin 10, RXD) -> FC UARTx TX
    Pi GND (pin 6)          -> FC GND

FC side (ArduPilot):
    Set SERIALx_PROTOCOL = 2 (MAVLink2) on the UART you wired up,
    and SERIALx_BAUD to match BAUD below.

Install:
    pip install pymavlink msgpack --break-system-packages
    (msgpack is optional - falls back to plain JSON if not installed)

--------------------------------------------------------------------------
THREE WAYS TO MOVE DATA (pick per use case, mix freely):

1. STANDARD MAVLINK MESSAGES  (preferred whenever one fits)
   e.g. NAMED_VALUE_FLOAT for a scalar, SET_POSITION_TARGET_LOCAL_NED for
   a velocity command, DISTANCE_SENSOR for a rangefinder reading, etc.
   ArduPilot parses these natively - no custom decode logic needed, and
   ground stations will display them for free.

2. GENERIC BLOB CHANNEL  (for arbitrary structured data)
   Uses MAVLink's built-in DATA96 message (96 raw bytes per packet) with a
   small header: [tag, format, seq, total]. `tag` is a number you choose to
   identify the "kind" of data (like a topic name). Payloads over 92 bytes
   are chunked and reassembled automatically. Encoded as MessagePack
   (compact) or JSON (readable) - see send_blob().

3. CUSTOM MAVLINK DIALECT  (not implemented here)
   For a mature project, define your own .xml MAVLink dialect with
   purpose-built messages and generate a custom pymavlink module. Most
   robust and efficient long-term, but more setup than most projects need
   up front. Worth revisiting once you know what data you're really moving.
--------------------------------------------------------------------------

Add a new data type by:
    - writing a small handler function, and
    - calling link.register_blob_handler(TAG, handler)
No changes to the core class needed.
"""

import json
import struct
import time

from pymavlink import mavutil

try:
    import msgpack
    HAVE_MSGPACK = True
except ImportError:
    HAVE_MSGPACK = False

PORT = "/dev/serial0"
BAUD = 115200


class FCLink:
    def __init__(self, port=PORT, baud=BAUD):
        self.master = mavutil.mavlink_connection(port, baud=baud)
        print("Waiting for heartbeat...")
        self.master.wait_heartbeat()
        print(f"Heartbeat received (system {self.master.target_system}, "
              f"component {self.master.target_component})")

        self.blob_handlers = {}   # tag(int) -> callback(obj)
        self.message_handlers = {}  # mavlink type name -> callback(msg)
        self._chunks = {}         # tag -> {seq: bytes} reassembly buffer

    # ---- tier 2: generic blob channel --------------------------------

    def register_blob_handler(self, tag, callback):
        """callback(obj) is called with the decoded python object whenever
        a complete blob with this tag arrives."""
        self.blob_handlers[tag] = callback

    def send_blob(self, tag, obj, use_msgpack=True):
        """Serialize `obj` and send as one or more DATA96 messages.
        tag: small int (0-255) you define to identify the data's meaning.
        """
        if use_msgpack and HAVE_MSGPACK:
            payload = msgpack.packb(obj, use_bin_type=True)
            fmt = 1  # 1 = msgpack
        else:
            payload = json.dumps(obj).encode("utf-8")
            fmt = 0  # 0 = json

        chunk_size = 92  # 96 byte payload - 4 header bytes
        chunks = [payload[i:i + chunk_size] for i in range(0, len(payload), chunk_size)]
        if not chunks:
            chunks = [b""]
        total = len(chunks)

        for seq, chunk in enumerate(chunks):
            header = struct.pack("BBBB", tag, fmt, seq, total)
            data = (header + chunk).ljust(96, b"\x00")
            self.master.mav.data96_send(0, len(header) + len(chunk), data)

    def _handle_data96(self, msg):
        raw = bytes(msg.data[:msg.len])
        if len(raw) < 4:
            return
        tag, fmt, seq, total = struct.unpack("BBBB", raw[:4])
        chunk = raw[4:]

        buf = self._chunks.setdefault(tag, {})
        buf[seq] = chunk
        if len(buf) < total:
            return  # still waiting on more chunks

        payload = b"".join(buf[i] for i in range(total))
        del self._chunks[tag]

        try:
            if fmt == 1 and HAVE_MSGPACK:
                obj = msgpack.unpackb(payload, raw=False)
            else:
                obj = json.loads(payload.decode("utf-8"))
        except Exception as e:
            print(f"Failed to decode blob tag={tag}: {e}")
            return

        handler = self.blob_handlers.get(tag)
        if handler:
            handler(obj)
        else:
            print(f"Unhandled blob tag={tag}: {obj}")

    # ---- tier 1: standard MAVLink message helpers ----------------------

    def register_message_handler(self, mavlink_type, callback):
        """callback(msg) is called for every incoming message of this type,
        e.g. register_message_handler('DISTANCE_SENSOR', my_func)."""
        self.message_handlers[mavlink_type] = callback

    def send_named_value(self, name, value):
        """Send one scalar using the standard NAMED_VALUE_FLOAT message."""
        self.master.mav.named_value_float_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            name.encode("utf-8")[:10].ljust(10, b"\x00"),
            float(value),
        )

    def send_velocity_command(self, vx, vy, vz, yaw_rate=0.0):
        """Body-frame velocity command in m/s. Vehicle must be in GUIDED
        mode for this to take effect."""
        type_mask = 0b0000111111000111  # enable velocity + yaw_rate only
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            type_mask,
            0, 0, 0,      # position (ignored)
            vx, vy, vz,   # velocity
            0, 0, 0,      # acceleration (ignored)
            0, yaw_rate,  # yaw, yaw_rate
        )

    # ---- main loop -------------------------------------------------------

    def spin(self):
        """Blocking loop: read messages and dispatch to handlers."""
        while True:
            msg = self.master.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue
            mtype = msg.get_type()

            if mtype == "DATA96":
                self._handle_data96(msg)
            elif mtype in self.message_handlers:
                self.message_handlers[mtype](msg)
            elif mtype == "HEARTBEAT":
                pass  # keepalive, nothing to do
            # unhandled types are silently ignored - register a handler
            # for anything you want to act on


# ===========================================================================
# Example usage - replace with your own sensors / decision logic
# ===========================================================================

TAG_SENSOR_READING = 1
TAG_DECISION_CMD = 2


def on_sensor_reading(obj):
    print(f"[FC->Pi] sensor blob: {obj}")


def on_named_value(msg):
    print(f"[FC->Pi] {msg.name.strip(chr(0))} = {msg.value}")


if __name__ == "__main__":
    link = FCLink()

    # tier 2: register handlers for custom data types
    link.register_blob_handler(TAG_SENSOR_READING, on_sensor_reading)

    # tier 1: register a handler for a standard MAVLink message type
    link.register_message_handler("NAMED_VALUE_FLOAT", on_named_value)

    # example: send a Pi-side decision as structured data
    link.send_blob(TAG_DECISION_CMD, {
        "action": "avoid_obstacle",
        "confidence": 0.87,
        "detected_at": time.time(),
    })

    # example: simple scalar telemetry via a standard message
    link.send_named_value("cpu_temp", 54.2)

    # example: an actual flight command (only takes effect in GUIDED mode)
    # link.send_velocity_command(vx=0.5, vy=0.0, vz=0.0)

    link.spin()