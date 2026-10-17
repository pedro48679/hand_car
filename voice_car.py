#!/usr/bin/env python3
"""
voice_car.py  --  Voice Controlled RC Car (Bluetooth / Serial)
==============================================================

Listens to spoken commands via microphone and sends movement
characters ('F', 'B', 'L', 'R', 'S') to the Arduino over Bluetooth.

Voice Commands recognized:
  - "forward", "front", "go", "ahead", "go"  -> F (Forward)
  - "back", "backward", "reverse"             -> B (Backward)
  - "left", "turn left"                       -> L (Turn Left)
  - "right", "turn right"                     -> R (Turn Right)
  - "stop", "halt", "wait", "brake"           -> S (Stop)

Usage:
  # With Arduino on Bluetooth COM port:
  python voice_car.py --port COM5

  # Dry-run / test mode (microphone only, no Arduino):
  python voice_car.py --no-serial
"""

import argparse
import logging
import sys
import time
import threading
import speech_recognition as sr

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SERIAL_BAUD      = 9600
DRIVE_DURATION_S = 1.00  # Forward ('F') and Backward ('B') drive for 1.0 s, then stop
TURN_DURATION_S  = 0.50  # Turns ('L', 'R') pulse for 500 ms, then stop

# Mapping spoken words / phrases to Arduino commands
COMMAND_MAP = {
    # Forward (primary command: "go")
    "forward": "F",
    "front": "F",
    "step": "F",
    "ahead": "F",
    "straight": "F",

    # Backward
    "back": "B",
    "backward": "B",
    "backwards": "B",
    "reverse": "B",

    # Left
    "left": "L",

    # Right
    "right": "R",

    # Stop
    "stop": "S",
    "halt": "S",
    "wait": "S",
    "brake": "S",
}

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('voice_car')


# ===========================================================================
# ARDUINO SERIAL SENDER
# ===========================================================================
class ArduinoSender:
    def __init__(self, port, baud, dry_run=False):
        self._dry_run = dry_run
        self._ser = None
        if dry_run:
            log.info('DRY-RUN mode: no serial port will be opened')
            return
        import serial
        try:
            self._ser = serial.Serial(port, baud, timeout=0)
            log.info('Serial port opened: %s @ %d baud', port, baud)
        except Exception as exc:
            log.error('Cannot open %s: %s', port, exc)
            log.error('Tip: run with --no-serial to test without Arduino connected')
            sys.exit(1)

    def send(self, cmd: str):
        """Transmit one ASCII command character to Arduino."""
        if self._dry_run:
            log.info('[DRY-RUN] Would send: %s', cmd)
            return
        if self._ser and self._ser.is_open:
            try:
                self._ser.write(cmd.encode('ascii'))
                log.info('Sent to Arduino: %s', cmd)
            except Exception as exc:
                log.warning('Serial write error: %s', exc)

    def close(self):
        if self._ser and self._ser.is_open:
            self.send('S')  # Guaranteed safe stop on exit
            time.sleep(0.05)
            self._ser.close()
            log.info('Serial port closed')


# ===========================================================================
# VOICE RECOGNITION LOOP
# ===========================================================================
def parse_spoken_text(text: str):
    """
    Scans the spoken sentence for movement keywords.
    Prioritizes 'stop' if mentioned.
    Returns: ('F' | 'B' | 'L' | 'R' | 'S' | None, matched_word)
    """
    words = text.lower().replace('-', ' ').split()

    # Safety first: check if user uttered any stop words anywhere in phrase
    for w in words:
        if COMMAND_MAP.get(w) == 'S':
            return 'S', w

    # Check words from newest/last spoken to first
    for w in reversed(words):
        if w in COMMAND_MAP:
            return COMMAND_MAP[w], w

    return None, None


def main():
    p = argparse.ArgumentParser(
        description='Voice Controlled RC Car',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--port', default=None,
        help='Bluetooth COM port (e.g. COM5 on Windows).',
    )
    p.add_argument(
        '--no-serial', action='store_true',
        help='Run in dry-run mode without sending to Arduino.',
    )
    p.add_argument(
        '--energy-threshold', type=int, default=300,
        help='Mic sensitivity threshold (lower = more sensitive, default: 300).',
    )

    args = p.parse_args()

    if not args.no_serial and not args.port:
        log.error('Must specify --port (e.g. --port COM5) or --no-serial for testing.')
        sys.exit(1)

    sender = ArduinoSender(port=args.port, baud=SERIAL_BAUD, dry_run=args.no_serial)

    move_timer = None
    lock = threading.Lock()

    def auto_stop(action_label):
        with lock:
            log.info("%s duration complete -> stopping car", action_label)
            sender.send('S')

    try:
        with sr.Microphone() as source:
            log.info("Calibrating microphone for ambient background noise (1 sec)...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            log.info("Ready! Speak commands: 'go', 'back', 'left', 'right', 'stop'...")
            log.info("Press Ctrl+C to stop.")

            while True:
                try:
                    audio = recognizer.listen(source, timeout=None, phrase_time_limit=3.0)
                    # Use Google's lightweight free recognition endpoint
                    text = recognizer.recognize_google(audio).lower()
                    log.info("Heard: '%s'", text)

                    cmd, matched_word = parse_spoken_text(text)
                    if cmd:
                        action_names = {
                            'F': 'GO (FORWARD 1.0s)',
                            'B': 'BACKWARD (1.0s)',
                            'L': 'TURN LEFT (0.5s)',
                            'R': 'TURN RIGHT (0.5s)',
                            'S': 'STOP',
                        }
                        log.info("--> Action [%s]: %s (from '%s')", cmd, action_names.get(cmd, cmd), matched_word)

                        with lock:
                            # Cancel any ongoing movement timer immediately
                            if move_timer and move_timer.is_alive():
                                move_timer.cancel()
                                move_timer = None

                            if cmd == 'S':
                                # Stop has highest priority: immediately stops the car
                                sender.send('S')

                            elif cmd in ('F', 'B'):
                                # Forward / Backward timed pulse for DRIVE_DURATION_S (1.0s)
                                sender.send(cmd)
                                move_timer = threading.Timer(
                                    DRIVE_DURATION_S, auto_stop, args=[action_names[cmd]]
                                )
                                move_timer.daemon = True
                                move_timer.start()

                            elif cmd in ('L', 'R'):
                                # Turn Left / Right timed pulse for TURN_DURATION_S (0.5s)
                                sender.send(cmd)
                                move_timer = threading.Timer(
                                    TURN_DURATION_S, auto_stop, args=[action_names[cmd]]
                                )
                                move_timer.daemon = True
                                move_timer.start()

                    else:
                        log.info("No matching driving command in: '%s'", text)

                except sr.UnknownValueError:
                    # Speech was unintelligible or background noise
                    pass
                except sr.RequestError as e:
                    log.warning("Could not request results from STT service: %s", e)

    except KeyboardInterrupt:
        log.info("Exiting on user interrupt...")
    finally:
        with lock:
            if move_timer and move_timer.is_alive():
                move_timer.cancel()
        sender.send('S')
        sender.close()
        log.info("Shutdown complete.")


if __name__ == '__main__':
    main()
