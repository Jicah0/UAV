#!/usr/bin/env python3
"""
Modular pymavlink companion-computer link: Raspberry Pi <-> ArduPilot FC
over UART. Standard MAVLink messages only - no custom channel.

Wiring:
    Pi GPIO14 (pin 8, TXD)  -> FC UARTx RX
    Pi GPIO15 (pin 10, RXD) -> FC UARTx TX
    Pi GND (pin 6)          -> FC GND

FC side (ArduPilot):
    Set SERIALx_PROTOCOL = 2 (MAVLink2) on the UART you wired up,
    and SERIALx_BAUD to match BAUD below.
    For send_landing_target(): also set PLND_ENABLED=1, PLND_TYPE=1.

Install:
    pip install pymavlink --break-system-packages

--------------------------------------------------------------------------
This uses only standard, built-in MAVLink message types (NAMED_VALUE_FLOAT,
ATTITUDE, SET_POSITION_TARGET_LOCAL_NED, LANDING_TARGET, etc.) - no custom
framing or chunking. ArduPilot parses these natively and any ground
station will display them for free.

Add a new message type to react to by:
    - writing a small handler function, and
    - calling link.register_message_handler("MESSAGE_TYPE_NAME", handler)
No changes to the core class needed.
--------------------------------------------------------------------------
"""

import time
from pymavlink import mavutil

PORT = "/dev/serial0"
BAUD = 115200


class FCLink:
    def __init__(self, port=PORT, baud=BAUD):
        self.master = mavutil.mavlink_connection(port, baud=baud)
        print("Waiting for heartbeat...")
        self.master.wait_heartbeat()
        print(f"Heartbeat received (system {self.master.target_system}, "
              f"component {self.master.target_component})")

        self.message_handlers = {}  # mavlink type name -> callback(msg)

    # ---- registering handlers -------------------------------------------

    def register_message_handler(self, mavlink_type, callback):
        """callback(msg) is called for every incoming message of this type,
        e.g. register_message_handler('ATTITUDE', my_func)."""
        self.message_handlers[mavlink_type] = callback

    # ---- standard MAVLink send helpers ------------------------------------

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
        # NOTE: as in the original script, this mask (the classic
        # DroneKit velocity-only mask) sets bit 11 = ignore, meaning
        # yaw_rate below is currently NOT applied by the FC regardless of
        # value. Clear bit 11 (mask = 0b0000011111000111) if you want it
        # to actually take effect.
        type_mask = 0b0000011111000111  # enable velocity + yaw_rate only
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,      # Drone-Centric Coordinates
            type_mask,
            0, 0, 0,      # position (ignored)
            vx, vy, vz,   # velocity
            0, 0, 0,      # acceleration (ignored)
            0, yaw_rate,  # yaw, yaw_rate
        )

    def send_landing_target(self, angle_x, angle_y, distance,
                             size_x=0.0, size_y=0.0, target_num=0,
                             frame=None, x=0.0, y=0.0, z=0.0,
                             q=(1.0, 0.0, 0.0, 0.0), target_type=2,
                             position_valid=0):
        """Send a MAVLink LANDING_TARGET message for ArduPilot Precision
        Landing/Loiter (requires PLND_ENABLED=1, PLND_TYPE=1 on the FC).

        Defaults to image-relative angle/distance mode (position_valid=0),
        which only needs angle_x, angle_y, distance, size_x, size_y - the
        fields a fiducial detector (e.g. AprilTag) naturally produces, and
        sidesteps having to rotate the detector's camera-frame pose into
        vehicle body FRD.

        Pass position_valid=1 with x/y/z/q populated instead (in
        MAV_FRAME_BODY_FRD) if you have a full 6-DOF body-frame pose to
        send.

        target_type defaults to 2 = LANDING_TARGET_TYPE_VISION_FIDUCIAL,
        the correct value for AprilTag/ArUco-style markers.
        """
        if frame is None:
            frame = mavutil.mavlink.MAV_FRAME_BODY_FRD

        self.master.mav.landing_target_send(
            int(time.time() * 1e6),        # time_usec
            target_num,                     # target_num
            frame,                          # frame
            float(angle_x),                 # angle_x, rad
            float(angle_y),                 # angle_y, rad
            float(distance),                # distance, m
            float(size_x),                  # size_x, rad
            float(size_y),                  # size_y, rad
            float(x), float(y), float(z),   # x, y, z, m
            list(q),                        # quaternion (w, x, y, z)
            target_type,                     # LANDING_TARGET_TYPE
            position_valid,                  # position_valid
        )

    # ---- main loop ---------------------------------------------------------

    def spin(self):
        """Blocking loop: read messages and dispatch to handlers."""
        while True:
            msg = self.master.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue
            mtype = msg.get_type()

            if mtype in self.message_handlers:
                self.message_handlers[mtype](msg)
            elif mtype == "HEARTBEAT":
                pass  # keepalive, nothing to do
            # unhandled types are silently ignored - register a handler
            # for anything you want to act on


# ===========================================================================
# Example usage - replace with your own logic
# ===========================================================================

def on_attitude(msg):
    print(f"ATTITUDE: roll={msg.roll:.2f} rad  pitch={msg.pitch:.2f} rad  "
          f"yaw={msg.yaw:.2f} rad")


def on_position(msg):
    lat = msg.lat / 1e7
    lon = msg.lon / 1e7
    alt = msg.relative_alt / 1000.0
    print(f"POSITION: lat={lat:.6f}  lon={lon:.6f}  alt={alt:.1f} m")


def on_sys_status(msg):
    print(f"BATTERY: {msg.voltage_battery / 1000.0:.2f} V  "
          f"{msg.battery_remaining}% remaining")


def on_named_value(msg):
    print(f"NAMED_VALUE: {msg.name.strip(chr(0))} = {msg.value}")


if __name__ == "__main__":
    link = FCLink()

    link.register_message_handler("ATTITUDE", on_attitude)
    link.register_message_handler("GLOBAL_POSITION_INT", on_position)
    link.register_message_handler("SYS_STATUS", on_sys_status)
    link.register_message_handler("NAMED_VALUE_FLOAT", on_named_value)

    # example: simple scalar telemetry via a standard message
    link.send_named_value("cpu_temp", 54.2)

    # example: an actual flight command (only takes effect in GUIDED mode)
    # link.send_velocity_command(vx=0.5, vy=0.0, vz=0.0)

    link.spin()