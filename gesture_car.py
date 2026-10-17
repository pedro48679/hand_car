#!/usr/bin/env python3
"""
gesture_car.py  --  Hand Gesture RC Car Controller
===================================================

Pipeline:
  [Phone Camera] -> HTTP/MJPEG stream (IP Webcam app, Android)
  [OpenCV]       -> Threaded reader, always yields the freshest frame
  [MediaPipe]    -> 21 hand landmarks, lite model (~15 ms/frame on CPU)
  [Classifier]   -> Maps gesture -> command  F / B / L / R / S
  [PySerial]     -> Sends 1 ASCII byte over Bluetooth COM port to Arduino

Gesture map:
  Open palm  facing camera  -> F  (Forward)
  Back of hand facing camera -> B  (Backward)
  Closed fist (0-1 fingers)  -> S  (Stop)
  Wrist shifted LEFT         -> L  (Turn left)
  Wrist shifted RIGHT        -> R  (Turn right)

Usage:
  # Full pipeline (phone stream + Arduino):
  python gesture_car.py --stream http://192.168.1.85:8080/video --port COM5

  # Test gesture detection only (PC webcam, no Arduino):
  python gesture_car.py --no-serial

  # Phone stream, no Arduino:
  python gesture_car.py --stream http://192.168.1.85:8080/video --no-serial
"""

import argparse
import sys
import time
import os
import math
import urllib.request
import threading
import logging

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ---------------------------------------------------------------------------
# CONFIGURATION  (edit these to tune behaviour)
# ---------------------------------------------------------------------------
WATCHDOG_INTERVAL_S  = 0.30   # Resend current command every N seconds
                               # so Arduino watchdog (500 ms) never fires
SMOOTHING_FRAMES     = 4      # Frames a gesture must persist before committing
TILT_THRESHOLD_DEG   = 40.0   # Hand angle from vertical: <= 45° = F/B, > 45° = L/R
CAM_WIDTH            = 640
CAM_HEIGHT           = 480
SERIAL_BAUD          = 9600   # Must match Arduino Serial.begin()
MIN_DETECTION_CONF   = 0.75
MIN_TRACKING_CONF    = 0.70

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('gesture_car')


# ===========================================================================
# STEP 1 -- PHONE CAMERA STREAM
# Zero-lag threaded MJPEG / RTSP reader.
# The background thread always reads the latest frame from the buffer,
# preventing OpenCV's internal queue from accumulating stale frames.
#
# Compatible sources:
#   IP Webcam (Android): http://<ip>:8080/video
#   DroidCam:            http://<ip>:4747/video
#   PC webcam:           0   (integer)
# ===========================================================================
class RealtimeCameraStream:
    def __init__(self, src):
        log.info('Opening stream: %s', src)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open stream: {src}')

        # Minimise OpenCV internal buffer to 1 frame -> eliminates built-in lag
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        self._lock    = threading.Lock()
        self._frame   = None
        self._ok      = False
        self._stopped = False

        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self):
        while not self._stopped:
            ok, frame = self.cap.read()
            with self._lock:
                self._ok    = ok
                self._frame = frame
            time.sleep(0.005)   # ~200 reads/s max; well above any camera FPS

    def read(self):
        """Return (ok, frame) -- always the freshest available frame."""
        with self._lock:
            return self._ok, (None if self._frame is None else self._frame.copy())

    def stop(self):
        self._stopped = True
        time.sleep(0.05)
        self.cap.release()


# ===========================================================================
# STEP 2 -- GESTURE CLASSIFIER  (MediaPipe Hands, lite model)
#
# Relevant landmark indices (0-indexed, 21 total):
#   0   WRIST
#   4   THUMB_TIP
#   5   INDEX_FINGER_MCP  (knuckle base)
#   6   INDEX_FINGER_PIP  (middle joint)
#   8   INDEX_FINGER_TIP
#   10  MIDDLE_FINGER_PIP
#   12  MIDDLE_FINGER_TIP
#   14  RING_FINGER_PIP
#   16  RING_FINGER_TIP
#   18  PINKY_PIP
#   20  PINKY_TIP
# ===========================================================================
class GestureClassifier:
    # Tip / PIP pairs for the four fingers (not thumb)
    _FINGER_TIPS  = [8,  12, 16, 20]
    _FINGER_PIPS  = [6,  10, 14, 18]

    # Hand landmark connections for skeleton rendering
    _HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
    ]

    MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'

    def __init__(self, model_file='gesture_recognizer.task'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(script_dir, model_file)
        if not os.path.exists(target_path):
            cwd_path = os.path.join(os.getcwd(), model_file)
            if os.path.exists(cwd_path):
                target_path = cwd_path
            else:
                log.info('Downloading MediaPipe model to %s ...', target_path)
                req = urllib.request.Request(self.MODEL_URL, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as resp, open(target_path, 'wb') as out_f:
                    out_f.write(resp.read())
                log.info('Model downloaded successfully.')

        # Read binary buffer to prevent Windows Unicode path issues in C++ Bazel runfiles
        with open(target_path, 'rb') as f:
            model_data = f.read()

        base_opts = mp_python.BaseOptions(model_asset_buffer=model_data)
        options = mp_vision.GestureRecognizerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=MIN_DETECTION_CONF,
            min_hand_presence_confidence=MIN_DETECTION_CONF,
            min_tracking_confidence=MIN_TRACKING_CONF,
        )
        self._recognizer = mp_vision.GestureRecognizer.create_from_options(options)

    def classify(self, frame):
        """
        Classify one BGR frame.

        Returns
        -------
        command : str or None
            'F', 'B', 'L', 'R', 'S' -- or None if no hand detected.
        annotated : np.ndarray
            Frame with landmark skeleton overlaid.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize(mp_image)

        annotated = frame.copy()
        h, w, _ = annotated.shape

        if not result.hand_landmarks:
            return None, annotated       # No hand in frame

        lm = result.hand_landmarks[0]    # list of 21 NormalizedLandmarks (x, y, z all 0-1)
        handedness = result.handedness[0][0].category_name if result.handedness else 'Right'

        # Draw skeleton connections
        points = [(int(p.x * w), int(p.y * h)) for p in lm]
        for p1, p2 in self._HAND_CONNECTIONS:
            cv2.line(annotated, points[p1], points[p2], (0, 255, 128), 2)
        # Draw landmark nodes
        for idx, pt in enumerate(points):
            color = (0, 0, 255) if idx in self._FINGER_TIPS or idx == 4 else (255, 200, 0)
            cv2.circle(annotated, pt, 4, color, -1)

        # ------------------------------------------------------------------
        # Feature A: count extended fingers
        # Uses both orientation-independent Euclidean distance from wrist
        # and vertical check to ensure robust detection at any angle.
        # ------------------------------------------------------------------
        def is_extended(tip, pip):
            dist_tip_sq = (lm[tip].x - lm[0].x)**2 + (lm[tip].y - lm[0].y)**2
            dist_pip_sq = (lm[pip].x - lm[0].x)**2 + (lm[pip].y - lm[0].y)**2
            return (dist_tip_sq > dist_pip_sq * 1.05) or (lm[tip].y < lm[pip].y)

        fingers_up = sum(
            1 for tip, pip in zip(self._FINGER_TIPS, self._FINGER_PIPS)
            if is_extended(tip, pip)
        )

        # ------------------------------------------------------------------
        # Feature B: hand tilt angle (degrees from vertical)
        # Vector from wrist (lm[0]) to middle knuckle (lm[9]).
        # - 0°   = pointing straight up
        # - < 0° = tilted to the left  (-90° is horizontal left)
        # - > 0° = tilted to the right (+90° is horizontal right)
        # ------------------------------------------------------------------
        dx_hand = lm[9].x - lm[0].x
        dy_hand = lm[9].y - lm[0].y
        tilt_deg = math.degrees(math.atan2(dx_hand, -dy_hand))

        # ------------------------------------------------------------------
        # Feature C: palm vs back orientation
        # Calculated via cross-product of palm triangle (Wrist -> Index, Wrist -> Pinky)
        # ------------------------------------------------------------------
        v_ax = lm[5].x - lm[0].x
        v_ay = lm[5].y - lm[0].y
        v_bx = lm[17].x - lm[0].x
        v_by = lm[17].y - lm[0].y
        cross_z = (v_ax * v_by) - (v_ay * v_bx)

        if handedness == 'Right':
            palm_facing = (cross_z > 0)
        else:
            palm_facing = (cross_z < 0)

        facing_label = 'Front (Palm)' if palm_facing else 'Back of Hand'

        # Draw hand state diagnostic info near the wrist
        info_text = f'{handedness} | Up:{fingers_up} | Tilt:{tilt_deg:+.0f}deg | {facing_label}'
        wrist_px = (max(10, min(w - 280, int(lm[0].x * w))), min(h - 15, int(lm[0].y * h) + 25))
        cv2.putText(annotated, info_text, wrist_px,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

        # ------------------------------------------------------------------
        # Decision tree:
        # - Closed hand / fist (<= 1 finger up) -> STOP ('S')
        # - Open hand (>= 3 fingers up):
        #     - abs(tilt) <= 45°:
        #         - Back of open hand   -> FORWARD ('F')
        #         - Front (palm) open   -> BACKWARD ('B')
        #     - tilt < -45°:            -> TURN LEFT ('L')
        #     - tilt > +45°:            -> TURN RIGHT ('R')
        # - Intermediate / 2 fingers   -> STOP ('S')
        # ------------------------------------------------------------------
        if fingers_up <= 1:
            command = 'S'
        elif fingers_up >= 3:
            if abs(tilt_deg) <= TILT_THRESHOLD_DEG:
                if palm_facing:
                    command = 'B'    # Front of hand (Palm) -> Backward
                else:
                    command = 'F'    # Back of hand -> Forward
            elif -135.0 <= tilt_deg < -TILT_THRESHOLD_DEG:
                command = 'L'        # Tilted left > 45° -> Turn Left
            elif TILT_THRESHOLD_DEG < tilt_deg <= 135.0:
                command = 'R'        # Tilted right > 45° -> Turn Right
            else:
                command = 'S'        # Hand pointing downwards -> Stop
        else:
            command = 'S'            # Intermediate transition state (e.g. 2 fingers)

        return command, annotated

    def close(self):
        self._recognizer.close()


# ===========================================================================
# STEP 3 -- ARDUINO SERIAL SENDER  (Bluetooth virtual COM port)
# ===========================================================================
class ArduinoSender:
    def __init__(self, port, baud, dry_run=False):
        self._dry_run = dry_run
        self._ser     = None
        if dry_run:
            log.info('DRY-RUN mode: no serial port will be opened')
            return
        import serial
        try:
            self._ser = serial.Serial(port, baud, timeout=0)
            log.info('Serial port opened: %s @ %d baud', port, baud)
        except Exception as exc:
            log.error('Cannot open %s: %s', port, exc)
            log.error('Tip: run with --no-serial to test without Arduino')
            sys.exit(1)

    def send(self, cmd: str):
        """Transmit one ASCII command character to the Arduino."""
        if self._dry_run:
            return
        if self._ser and self._ser.is_open:
            try:
                self._ser.write(cmd.encode('ascii'))
            except Exception as exc:
                log.warning('Serial write error: %s', exc)

    def close(self):
        if self._ser and self._ser.is_open:
            self.send('S')       # Guaranteed stop on exit
            time.sleep(0.05)
            self._ser.close()
            log.info('Serial port closed')


# ===========================================================================
# STEP 4 -- MAIN CONTROL LOOP
# ===========================================================================
def run(args):
    # -- Step 1: open camera
    src = 0 if args.stream is None else args.stream
    camera = RealtimeCameraStream(src)
    time.sleep(0.5)    # Let background thread fill the first frame

    # -- Step 2: init classifier
    classifier = GestureClassifier()

    # -- Step 3: init sender
    sender = ArduinoSender(
        port    = args.port or 'COM_PLACEHOLDER',
        baud    = SERIAL_BAUD,
        dry_run = args.no_serial,
    )

    # -- State machine
    committed_cmd   = 'S'    # Last command actually sent to Arduino
    candidate_cmd   = 'S'    # Current raw gesture candidate
    candidate_count = 0      # Consecutive frames with same candidate
    last_sent_time  = 0.0    # monotonic time of last serial send

    log.info('Controller running  --  press Q in preview window to quit')

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            # ---- CLASSIFY ------------------------------------------------
            raw_cmd, annotated = classifier.classify(frame)
            if raw_cmd is None:
                raw_cmd = 'S'    # No hand -> stop

            # ---- SMOOTH --------------------------------------------------
            if raw_cmd == candidate_cmd:
                candidate_count += 1
            else:
                candidate_cmd   = raw_cmd
                candidate_count = 1

            # Snappy safety stop: stop commits in 2 frames; F/B use full SMOOTHING_FRAMES
            required_frames = 2 if candidate_cmd == 'S' else SMOOTHING_FRAMES
            new_cmd = candidate_cmd if candidate_count >= required_frames else committed_cmd

            # ---- SEND TO ARDUINO -----------------------------------------
            now          = time.monotonic()
            changed      = new_cmd != committed_cmd
            watchdog_due = (now - last_sent_time) >= WATCHDOG_INTERVAL_S

            if changed or watchdog_due:
                sender.send(new_cmd)
                last_sent_time = now
                if changed:
                    log.info('Command: %s -> %s', committed_cmd, new_cmd)
                    committed_cmd = new_cmd

            # ---- DEBUG OVERLAY -------------------------------------------
            color_map = {
                'F': (0, 180, 0),      # Green for Forward
                'B': (0, 100, 220),    # Orange/Red for Backward
                'L': (220, 140, 0),    # Cyan/Blue for Turn Left
                'R': (200, 0, 200),    # Magenta/Purple for Turn Right
                'S': (60, 60, 60),     # Dark Gray for Stop
            }
            h, w = annotated.shape[:2]
            bar_col = color_map.get(committed_cmd, (120, 120, 120))
            cv2.rectangle(annotated, (0, 0), (w, 54), bar_col, -1)

            action_desc = {
                'F': 'FORWARD (Back of Hand)',
                'B': 'BACKWARD (Front / Palm)',
                'L': 'TURN LEFT (Tilted Left)',
                'R': 'TURN RIGHT (Tilted Right)',
                'S': 'STOP (Closed Fist / Idle)',
            }.get(committed_cmd, 'STOP')

            label = (
                f'[{committed_cmd}] {action_desc}  |  '
                f'raw: {raw_cmd} ({candidate_count}/{required_frames})'
            )
            cv2.putText(annotated, label, (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

            cv2.imshow('Gesture Car Controller -- press Q to quit', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        log.info('Interrupted')
    finally:
        log.info('Shutting down...')
        sender.send('S')          # Always stop the car on exit
        time.sleep(0.1)
        camera.stop()
        classifier.close()
        sender.close()
        cv2.destroyAllWindows()
        log.info('Done.')


# ===========================================================================
# ENTRY POINT
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='Hand Gesture RC Car Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  # Full pipeline (phone + Arduino):\n'
            '  python gesture_car.py --stream http://192.168.1.85:8080/video --port COM5\n'
            '\n'
            '  # Gesture test only (PC webcam, no Arduino):\n'
            '  python gesture_car.py --no-serial\n'
            '\n'
            '  # Phone stream, no Arduino:\n'
            '  python gesture_car.py --stream http://192.168.1.85:8080/video --no-serial\n'
        ),
    )
    p.add_argument(
        '--stream', default=None,
        help=(
            'Phone camera stream URL '
            '(e.g. http://192.168.1.85:8080/video for IP Webcam). '
            'Omit to use the PC webcam (index 0).'
        ),
    )
    p.add_argument(
        '--port', default=None,
        help='Bluetooth COM port (e.g. COM5 on Windows, /dev/rfcomm0 on Linux/Mac).',
    )
    p.add_argument(
        '--no-serial', action='store_true', default=False,
        help='Dry-run: classify gestures and log commands but do not open serial.',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not args.no_serial and args.port is None:
        print('ERROR: provide --port COMx or use --no-serial for testing.')
        print('Run: python gesture_car.py --help')
        sys.exit(1)
    run(args)
