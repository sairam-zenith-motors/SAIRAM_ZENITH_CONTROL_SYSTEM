import math
import struct
import time

import can
import cv2
import numpy as np
import serial


# ============================================================
# USER SETTINGS
# ============================================================

# Camera device:
# 0 = /dev/video0
# 1 = /dev/video1
CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Keep at 0 during your current camera testing.
# Adjust only after mounting and calibration.
CAMERA_OFFSET_PX = 0

# Replace this with the actual measured distance
# between the left and right lane boundaries.
REAL_LANE_WIDTH_M = 3.0


# ============================================================
# CAN SETTINGS — STEP 17
# ============================================================

# Set False to run the vision pipeline without transmitting CAN.
CAN_ENABLED = True

# Linux SocketCAN interface name.
# Common value: can0
CAN_CHANNEL = "can0"

# Standard 11-bit CAN identifier for steering command.
STEERING_COMMAND_CAN_ID = 0x201

# Maximum CAN transmission rate.
CAN_SEND_RATE_HZ = 20.0

# Safe steering command when current lane data is invalid.
SAFE_STEERING_ON_LANE_LOSS_DEG = 0.0


# ============================================================
# SERIAL SETTINGS FOR ESP32 (direct USB, no CAN)
# ============================================================

# Set False to skip ESP32 serial entirely.
ESP32_ENABLED = True

# Confirm with: ls /dev/ttyUSB* /dev/ttyACM*
ESP32_PORT = "/dev/ttyACM0"

ESP32_BAUD = 115200


# ============================================================
# IMAGE-PROCESSING SETTINGS
# ============================================================

CANNY_LOW = 50
CANNY_HIGH = 150

HOUGH_THRESHOLD = 40
MIN_LINE_LENGTH = 40
MAX_LINE_GAP = 80

# Ignore nearly horizontal lines.
MIN_ABS_SLOPE = 0.3


# ============================================================
# STEERING-CONTROL SETTINGS
# ============================================================

CENTER_DEADBAND_PX = 5
HEADING_DEADBAND_DEG = 2.0

# Step 14 gains.
K1_LATERAL = 8.0
K2_HEADING = 0.7

# Step 15 steering limits.
MAX_LEFT_STEERING_DEG = -25.0
MAX_RIGHT_STEERING_DEG = 25.0

# Step 16 smoothing.
STEERING_FILTER_ALPHA = 0.2

STEERING_DEADBAND_DEG = 0.5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(value, minimum, maximum):
    """Keep value between minimum and maximum."""

    return max(minimum, min(maximum, value))


def get_line_x_at_y(line, target_y):
    """
    Calculate the x-coordinate where a line crosses target_y.

    line:
    (x1, y1, x2, y2)
    """

    x1, y1, x2, y2 = line

    # Horizontal line: cannot calculate another y intersection.
    if y2 == y1:
        return None

    return x1 + (
        (target_y - y1)
        * (x2 - x1)
        / (y2 - y1)
    )


def average_line_position(lines, target_y, image_width):
    """
    Calculate the average x-coordinate of valid lines
    at target_y.
    """

    positions = []

    for line in lines:
        x_position = get_line_x_at_y(line, target_y)

        if (
            x_position is not None
            and 0 <= x_position < image_width
        ):
            positions.append(x_position)

    if not positions:
        return None

    return int(np.mean(positions))


def open_can_bus():
    """
    Open the Linux SocketCAN interface.

    Returns:
        python-can Bus object, or None if CAN is disabled/unavailable.
    """

    if not CAN_ENABLED:
        print("CAN disabled. Running vision only.")
        return None

    try:
        bus = can.Bus(
            interface="socketcan",
            channel=CAN_CHANNEL
        )

        print(f"CAN opened successfully on {CAN_CHANNEL}.")
        return bus

    except (can.CanError, OSError) as error:
        print(f"Warning: CAN could not be opened: {error}")
        print("Vision will continue without CAN transmission.")
        return None


def build_steering_can_message(
    steering_angle_deg,
    command_valid,
    rolling_counter
):
    """
    Build one 8-byte standard CAN frame.

    CAN ID:
        0x201

    Payload:
        Bytes 0-1: signed steering angle in 0.01 degree
        Byte 2:     command-valid flag, 1 or 0
        Byte 3:     rolling counter, 0-255
        Bytes 4-7:  monotonic timestamp in milliseconds
    """

    safe_angle_deg = clamp(
        steering_angle_deg,
        MAX_LEFT_STEERING_DEG,
        MAX_RIGHT_STEERING_DEG
    )

    # Example: +5.20 degrees becomes signed integer +520.
    angle_centideg = int(round(safe_angle_deg * 100.0))

    valid_byte = 1 if command_valid else 0
    counter_byte = rolling_counter & 0xFF
    timestamp_ms = int(time.monotonic() * 1000.0) & 0xFFFFFFFF

    # Big-endian payload:
    # h = signed 16-bit, B = unsigned 8-bit, I = unsigned 32-bit.
    payload = struct.pack(
        ">hBBI",
        angle_centideg,
        valid_byte,
        counter_byte,
        timestamp_ms
    )

    return can.Message(
        arbitration_id=STEERING_COMMAND_CAN_ID,
        data=payload,
        is_extended_id=False
    )


def send_steering_can_message(
    bus,
    steering_angle_deg,
    command_valid,
    rolling_counter
):
    """Send one steering command through SocketCAN."""

    if bus is None:
        return False

    message = build_steering_can_message(
        steering_angle_deg=steering_angle_deg,
        command_valid=command_valid,
        rolling_counter=rolling_counter
    )

    try:
        bus.send(message, timeout=0.05)
        return True

    except can.CanError as error:
        print(f"CAN transmission error: {error}")
        return False


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_V4L2
)

if not cap.isOpened():
    print(f"Error: Camera index {CAMERA_INDEX} could not open.")
    print("Try CAMERA_INDEX = 1 or CAMERA_INDEX = 2.")
    raise SystemExit

print("Camera opened successfully.")
print("Press Q to close all windows.")


# ============================================================
# OPEN CAN
# ============================================================

can_bus = open_can_bus()


# ============================================================
# OPEN ESP32 SERIAL
# ============================================================

esp32 = None

if ESP32_ENABLED:
    try:
        esp32 = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=1)

        # Allow ESP32 to reset after the serial port opens.
        time.sleep(2)

        print(f"ESP32 serial opened on {ESP32_PORT}.")

    except Exception as error:
        print(f"Warning: ESP32 serial could not be opened: {error}")
        print("Vision will continue without ESP32 transmission.")
        esp32 = None
else:
    print("ESP32 serial disabled. Running vision only.")


# ============================================================
# PERSISTENT VALUES
# ============================================================

# Step 16 memory. This must remain outside the loop.
previous_filtered_angle_deg = 0.0

# Used to limit CAN transmission rate.
last_can_send_time = 0.0

# Rolling counter included in every valid CAN frame.
rolling_counter = 0

# Used only for displaying the latest CAN transmission status.
last_can_message = "CAN: no frame sent"


# ============================================================
# REAL-TIME PROCESSING LOOP
# ============================================================

try:
    while True:

        # ====================================================
        # STEP 1: CAPTURE LIVE CAMERA FRAME
        # ====================================================

        ret, frame = cap.read()

        if not ret or frame is None:
            print("Failed to read frame from camera.")
            break


        # ====================================================
        # STEP 2: RESIZE IMAGE
        # ====================================================

        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        frame_height, frame_width = frame.shape[:2]


        # ====================================================
        # STEP 10: VEHICLE / IMAGE CENTRE
        # ====================================================

        vehicle_center_x = (
            frame_width // 2
            + CAMERA_OFFSET_PX
        )

        vehicle_center_x = int(
            clamp(
                vehicle_center_x,
                0,
                frame_width - 1
            )
        )


        # ====================================================
        # STEP 3: SELECT LOWER-HALF ROI
        # ====================================================

        roi_start_y = frame_height // 2

        roi = frame[
            roi_start_y:frame_height,
            0:frame_width
        ]

        roi_height, roi_width = roi.shape[:2]


        # ====================================================
        # STEP 4: GRAYSCALE AND GAUSSIAN BLUR
        # ====================================================

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )


        # ====================================================
        # STEP 5: CANNY EDGE DETECTION
        # ====================================================

        edges = cv2.Canny(
            blur,
            threshold1=CANNY_LOW,
            threshold2=CANNY_HIGH
        )


        # ====================================================
        # STEP 6: HOUGH LINE DETECTION
        # ====================================================

        detected_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=HOUGH_THRESHOLD,
            minLineLength=MIN_LINE_LENGTH,
            maxLineGap=MAX_LINE_GAP
        )


        # ====================================================
        # STEP 7: SEPARATE LEFT AND RIGHT LINES
        # ====================================================

        left_lines = []
        right_lines = []

        if detected_lines is not None:

            for detected_line in detected_lines:

                x1, y1, x2, y2 = detected_line[0]

                # Skip vertical lines for slope calculation.
                if x2 == x1:
                    continue

                slope = (
                    (y2 - y1)
                    / (x2 - x1)
                )

                midpoint_x = (
                    x1 + x2
                ) / 2.0

                if (
                    slope < -MIN_ABS_SLOPE
                    and midpoint_x < vehicle_center_x
                ):
                    left_lines.append(
                        (x1, y1, x2, y2)
                    )

                elif (
                    slope > MIN_ABS_SLOPE
                    and midpoint_x > vehicle_center_x
                ):
                    right_lines.append(
                        (x1, y1, x2, y2)
                    )


        # ====================================================
        # STEP 8: FIND NEAR LANE POSITIONS
        # ====================================================

        near_y = roi_height - 20

        near_left_x = average_line_position(
            left_lines,
            near_y,
            roi_width
        )

        near_right_x = average_line_position(
            right_lines,
            near_y,
            roi_width
        )


        # ====================================================
        # STEP 9: CALCULATE LANE CENTRE
        # ====================================================

        lane_center_x = None

        if (
            near_left_x is not None
            and near_right_x is not None
            and near_right_x > near_left_x
        ):
            lane_center_x = int(
                (near_left_x + near_right_x) / 2
            )


        # ====================================================
        # STEP 11: CALCULATE LANE ERROR IN PIXELS
        # ====================================================

        lane_error_px = None
        lane_position_status = "NO LANE DATA"

        if lane_center_x is not None:

            lane_error_px = (
                lane_center_x
                - vehicle_center_x
            )

            if abs(lane_error_px) <= CENTER_DEADBAND_PX:
                lane_position_status = "VEHICLE CENTRED"

            elif lane_error_px > 0:
                lane_position_status = "LANE CENTRE RIGHT"

            else:
                lane_position_status = "LANE CENTRE LEFT"


        # ====================================================
        # STEP 12: CONVERT PIXEL ERROR TO METRES
        # ====================================================

        lane_width_px = None
        meters_per_pixel = None
        lateral_error_m = None

        if (
            near_left_x is not None
            and near_right_x is not None
            and lane_error_px is not None
        ):

            lane_width_px = (
                near_right_x
                - near_left_x
            )

            if lane_width_px > 0:

                meters_per_pixel = (
                    REAL_LANE_WIDTH_M
                    / lane_width_px
                )

                lateral_error_m = (
                    lane_error_px
                    * meters_per_pixel
                )


        # ====================================================
        # STEP 13: CALCULATE HEADING ERROR
        # ====================================================

        far_y = int(
            roi_height * 0.40
        )

        far_left_x = average_line_position(
            left_lines,
            far_y,
            roi_width
        )

        far_right_x = average_line_position(
            right_lines,
            far_y,
            roi_width
        )

        near_center_x = lane_center_x
        far_center_x = None

        if (
            far_left_x is not None
            and far_right_x is not None
            and far_right_x > far_left_x
        ):
            far_center_x = int(
                (far_left_x + far_right_x) / 2
            )

        heading_error_deg = None
        heading_status = "HEADING UNAVAILABLE"

        if (
            near_center_x is not None
            and far_center_x is not None
        ):

            dx = (
                far_center_x
                - near_center_x
            )

            dy = (
                near_y
                - far_y
            )

            if dy > 0:

                heading_error_rad = math.atan2(
                    dx,
                    dy
                )

                heading_error_deg = math.degrees(
                    heading_error_rad
                )

                if (
                    abs(heading_error_deg)
                    <= HEADING_DEADBAND_DEG
                ):
                    heading_status = "LANE STRAIGHT"

                elif heading_error_deg > 0:
                    heading_status = "LANE TURNS RIGHT"

                else:
                    heading_status = "LANE TURNS LEFT"


        # ====================================================
        # STEP 14: GENERATE RAW STEERING ANGLE
        # ====================================================

        lateral_component_deg = None
        heading_component_deg = None
        raw_steering_angle_deg = None

        if (
            lateral_error_m is not None
            and heading_error_deg is not None
        ):

            lateral_component_deg = (
                K1_LATERAL
                * lateral_error_m
            )

            heading_component_deg = (
                K2_HEADING
                * heading_error_deg
            )

            raw_steering_angle_deg = (
                lateral_component_deg
                + heading_component_deg
            )


        # ====================================================
        # STEP 15: LIMIT STEERING ANGLE
        # ====================================================

        limited_steering_angle_deg = None
        steering_limited = False

        if raw_steering_angle_deg is not None:

            limited_steering_angle_deg = clamp(
                raw_steering_angle_deg,
                MAX_LEFT_STEERING_DEG,
                MAX_RIGHT_STEERING_DEG
            )

            steering_limited = (
                abs(
                    limited_steering_angle_deg
                    - raw_steering_angle_deg
                ) > 0.001
            )


        # ====================================================
        # STEP 16: SMOOTH STEERING COMMAND
        # ====================================================

        lane_command_valid = (
            limited_steering_angle_deg is not None
        )

        if lane_command_valid:

            filtered_steering_angle_deg = (
                STEERING_FILTER_ALPHA
                * limited_steering_angle_deg
                + (1.0 - STEERING_FILTER_ALPHA)
                * previous_filtered_angle_deg
            )

        else:

            # Slowly return the displayed/filter value to zero.
            filtered_steering_angle_deg = (
                (1.0 - STEERING_FILTER_ALPHA)
                * previous_filtered_angle_deg
            )

        filtered_steering_angle_deg = clamp(
            filtered_steering_angle_deg,
            MAX_LEFT_STEERING_DEG,
            MAX_RIGHT_STEERING_DEG
        )

        previous_filtered_angle_deg = (
            filtered_steering_angle_deg
        )


        # ====================================================
        # FINAL STEERING DIRECTION
        # ====================================================

        if not lane_command_valid:
            final_steering_command = "NO VALID LANE"

        elif (
            filtered_steering_angle_deg
            > STEERING_DEADBAND_DEG
        ):
            final_steering_command = "STEER RIGHT"

        elif (
            filtered_steering_angle_deg
            < -STEERING_DEADBAND_DEG
        ):
            final_steering_command = "STEER LEFT"

        else:
            final_steering_command = "KEEP STRAIGHT"


        # ====================================================
        # STEP 17: SEND COMMAND THROUGH CAN
        # ====================================================

        # Send the filtered steering angle only when the current
        # lane command is valid. Otherwise send zero with valid=0.
        if lane_command_valid:
            can_command_angle_deg = filtered_steering_angle_deg
            can_command_valid = True
        else:
            can_command_angle_deg = SAFE_STEERING_ON_LANE_LOSS_DEG
            can_command_valid = False

        current_time = time.monotonic()
        can_send_interval = 1.0 / CAN_SEND_RATE_HZ

        if (
            current_time - last_can_send_time
            >= can_send_interval
        ):

            # Print steering angle in decimal and hexadecimal
            angle_centideg = int(round(can_command_angle_deg * 100.0))

            print("--------------------------------")
            print(f"Steering Angle : {can_command_angle_deg:.2f} deg")
            print(f"Decimal        : {angle_centideg}")
            print(f"Hex            : 0x{angle_centideg & 0xFFFF:04X}")

            # Build CAN message
            message = build_steering_can_message(
                can_command_angle_deg,
                can_command_valid,
                rolling_counter
            )

            # Print CAN frame
            print("CAN ID :", hex(message.arbitration_id))
            print("Bytes  :", " ".join(f"{b:02X}" for b in message.data))

            # Send CAN frame
            try:
                if can_bus is not None:
                    can_bus.send(message, timeout=0.05)
                    message_sent = True
                else:
                    message_sent = False

            except can.CanError as error:
                print(f"CAN transmission error: {error}")
                message_sent = False

            # Send steering angle to ESP32 over direct serial
            if esp32 is not None:
                try:
                    esp32.write(
                        f"{can_command_angle_deg:.2f}\n".encode()
                    )

                    if esp32.in_waiting:
                        response = esp32.readline().decode(
                            errors="ignore"
                        ).strip()

                        if response:
                            print(f"[ESP32] {response}")

                except Exception as error:
                    print(f"ESP32 serial error: {error}")

            if message_sent:
                last_can_message = (
                    f"CAN 0x{STEERING_COMMAND_CAN_ID:03X}: "
                    f"{can_command_angle_deg:+.2f} deg "
                    f"valid={int(can_command_valid)} "
                    f"count={rolling_counter}"
                )

                rolling_counter = (rolling_counter + 1) & 0xFF

            elif CAN_ENABLED:
                last_can_message = "CAN: not connected"
            else:
                last_can_message = "CAN: disabled"

            last_can_send_time = current_time


        # ====================================================
        # DRAW RESULT IMAGE
        # ====================================================

        result_image = roi.copy()


        # Left-lane candidates: blue.
        for x1, y1, x2, y2 in left_lines:

            cv2.line(
                result_image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3
            )


        # Right-lane candidates: red.
        for x1, y1, x2, y2 in right_lines:

            cv2.line(
                result_image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )


        # Near reference row: green.
        cv2.line(
            result_image,
            (0, near_y),
            (roi_width - 1, near_y),
            (0, 255, 0),
            2
        )


        # Far reference row: cyan.
        cv2.line(
            result_image,
            (0, far_y),
            (roi_width - 1, far_y),
            (255, 255, 0),
            2
        )


        # Vehicle/image centre: white.
        cv2.line(
            result_image,
            (vehicle_center_x, 0),
            (vehicle_center_x, roi_height - 1),
            (255, 255, 255),
            2
        )


        if near_left_x is not None:
            cv2.circle(
                result_image,
                (near_left_x, near_y),
                8,
                (255, 0, 0),
                -1
            )


        if near_right_x is not None:
            cv2.circle(
                result_image,
                (near_right_x, near_y),
                8,
                (0, 0, 255),
                -1
            )


        if near_center_x is not None:

            cv2.circle(
                result_image,
                (near_center_x, near_y),
                9,
                (0, 255, 255),
                -1
            )

            cv2.line(
                result_image,
                (vehicle_center_x, near_y),
                (near_center_x, near_y),
                (255, 0, 255),
                4
            )


        if far_left_x is not None:
            cv2.circle(
                result_image,
                (far_left_x, far_y),
                7,
                (255, 0, 0),
                -1
            )


        if far_right_x is not None:
            cv2.circle(
                result_image,
                (far_right_x, far_y),
                7,
                (0, 0, 255),
                -1
            )


        if far_center_x is not None:
            cv2.circle(
                result_image,
                (far_center_x, far_y),
                9,
                (0, 165, 255),
                -1
            )


        if (
            near_center_x is not None
            and far_center_x is not None
        ):
            cv2.line(
                result_image,
                (near_center_x, near_y),
                (far_center_x, far_y),
                (0, 165, 255),
                4
            )


        # ====================================================
        # CREATE DISPLAY TEXT
        # ====================================================

        pixel_error_text = (
            f"Pixel: {lane_error_px:+d}px"
            if lane_error_px is not None
            else "Pixel: unavailable"
        )

        lateral_error_text = (
            f"Lateral: {lateral_error_m:+.3f}m"
            if lateral_error_m is not None
            else "Lateral: unavailable"
        )

        heading_error_text = (
            f"Heading: {heading_error_deg:+.2f}deg"
            if heading_error_deg is not None
            else "Heading: unavailable"
        )

        raw_steering_text = (
            f"Raw: {raw_steering_angle_deg:+.2f}deg"
            if raw_steering_angle_deg is not None
            else "Raw: unavailable"
        )

        limited_steering_text = (
            f"Limited: {limited_steering_angle_deg:+.2f}deg"
            if limited_steering_angle_deg is not None
            else "Limited: unavailable"
        )

        filtered_steering_text = (
            f"Filtered: "
            f"{filtered_steering_angle_deg:+.2f}deg"
        )

        limit_status_text = (
            "LIMIT APPLIED"
            if steering_limited
            else "WITHIN LIMIT"
        )


        # ====================================================
        # DISPLAY TEXT
        # ====================================================

        text_items = [
            (pixel_error_text, (10, 18), (255, 0, 255)),
            (lateral_error_text, (10, 37), (255, 0, 255)),
            (heading_error_text, (10, 56), (0, 165, 255)),
            (raw_steering_text, (10, 75), (0, 255, 255)),
            (limited_steering_text, (10, 94), (0, 255, 0)),
            (filtered_steering_text, (10, 113), (255, 255, 0)),
            (final_steering_command, (10, 138), (0, 255, 0)),
            (limit_status_text, (10, 158), (0, 255, 255)),
            (last_can_message, (10, 178), (255, 255, 255)),
            (lane_position_status, (340, 18), (255, 0, 255)),
            (heading_status, (340, 37), (0, 165, 255))
        ]

        for text, position, colour in text_items:
            cv2.putText(
                result_image,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                colour,
                1
            )


        # ====================================================
        # FULL FRAME DISPLAY
        # ====================================================

        display_frame = frame.copy()

        cv2.line(
            display_frame,
            (vehicle_center_x, 0),
            (vehicle_center_x, frame_height - 1),
            (255, 255, 255),
            2
        )

        cv2.rectangle(
            display_frame,
            (0, roi_start_y),
            (frame_width - 1, frame_height - 1),
            (0, 255, 0),
            2
        )


        # ====================================================
        # SHOW WINDOWS
        # ====================================================

        cv2.imshow(
            "1 - Full Camera Frame",
            display_frame
        )

        cv2.imshow(
            "2 - Selected ROI",
            roi
        )

        cv2.imshow(
            "3 - Grayscale ROI",
            gray
        )

        cv2.imshow(
            "4 - Blurred ROI",
            blur
        )

        cv2.imshow(
            "5 - Canny Edge Image",
            edges
        )

        cv2.imshow(
            "6 to 17 - CAN Steering Command",
            result_image
        )


        # Press Q to close.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


finally:

    # ========================================================
    # SAFE CLEANUP
    # ========================================================

    # Send an invalid zero-angle command before closing CAN.
    if can_bus is not None:
        send_steering_can_message(
            bus=can_bus,
            steering_angle_deg=0.0,
            command_valid=False,
            rolling_counter=rolling_counter
        )

        time.sleep(0.05)
        can_bus.shutdown()
        print("CAN interface closed safely.")

    if esp32 is not None:
        esp32.close()
        print("ESP32 serial closed safely.")

    cap.release()
    cv2.destroyAllWindows()

    print("Camera closed successfully.")
