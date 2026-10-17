#!/usr/bin/env python3
"""
voice_car.py  --  Voice-Controlled RC Car with Timed Movements
==============================================================

Pipeline:
  [Microphone]   -> sounddevice (16 kHz 16-bit Mono PCM)
  [Vosk]         -> Real-time offline Kaldi speech recognition (<50ms latency)
  [Timer Engine] -> Executes timed, incremental movements & auto-stops
  [BLE / Serial] -> Transmits commands over Bluetooth Low Energy (BT24)
                    or Classic Bluetooth/USB Serial (9600 baud)

Strict Voice Commands:
  "Forward" -> Move Forward for 1.0s, then Auto-Stop
  "Back"    -> Move Backward for 1.0s, then Auto-Stop
  "Step"    -> Move Forward for 0.5s, then Auto-Stop
  "Left"    -> Turn Left for 0.5s, then Auto-Stop
  "Right"   -> Turn Right for 0.5s, then Auto-Stop
  "Stop"    -> Immediate Stop

Usage:
  # Default (Mac mic + Auto BLE to car BT24):
  python voice_car.py

  # Test voice commands without connecting to the car:
  python voice_car.py --no-serial

  # Custom durations (e.g. 0.4s turns, 0.8s forward/back, 0.3s step):
  python voice_car.py --move-duration 1.0 --step-duration 0.5 --turn-duration 0.5

  # Specify a particular microphone index:
  python voice_car.py --device 0
"""

import argparse
import sys
import time
import json
import queue
import logging

import sounddevice as sd
from vosk import Model, KaldiRecognizer

# Import Bluetooth / Serial communication from gesture_car
from gesture_car import ArduinoBleSender, ArduinoSender, SERIAL_BAUD, WATCHDOG_INTERVAL_S

# ---------------------------------------------------------------------------
# LOGGING & CONFIG
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('voice_car')

SAMPLE_RATE = 16000

# Default durations requested by user
DEFAULT_MOVE_DURATION_S = 1.0   # Forward and Back move for 1.0s
DEFAULT_STEP_DURATION_S = 0.5   # Step moves forward for 0.5s
DEFAULT_TURN_DURATION_S = 0.5   # Left and Right turn for 0.5s

# Strict vocabulary: ONLY these words exist in the acoustic/language model
GRAMMAR = [
    "forward",
    "back",
    "step",
    "left",
    "right",
    "stop",
    "[unk]"
]


# ===========================================================================
# VOICE RECOGNITION & TIMED MOVEMENT CONTROLLER
# ===========================================================================
class VoiceCarController:
    def __init__(
        self,
        sender,
        device=None,
        lang='en-us',
        move_duration=DEFAULT_MOVE_DURATION_S,
        step_duration=DEFAULT_STEP_DURATION_S,
        turn_duration=DEFAULT_TURN_DURATION_S,
    ):
        self.sender = sender
        self.device = device
        self.move_duration = move_duration
        self.step_duration = step_duration
        self.turn_duration = turn_duration

        self.audio_queue = queue.Queue()
        self.running = False
        self.current_cmd = 'S'
        self.last_sent_time = 0.0

        # Timing tracking for all movements (no continuous movement)
        self.action_deadline = None     # Monotonic timestamp when current timed action finishes
        self.action_desc = 'IDLE'
        self.utterance_consumed = False

        log.info('Loading Vosk speech model (%s) ...', lang)
        self.model = Model(lang=lang)
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE, json.dumps(GRAMMAR))
        self.recognizer.SetWords(False)
        log.info('Speech model ready.')

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            log.warning('Audio status: %s', status)
        self.audio_queue.put(bytes(indata))

    def run(self):
        self.running = True
        log.info('Starting audio stream on device: %s', self.device if self.device is not None else 'Default Mic')

        # Display instructions
        print('\n' + '=' * 65)
        print('  STRICT VOICE COMMANDS:')
        print(f'    "Forward"  -> Move Forward for {self.move_duration:.1f}s (Auto-Stop)')
        print(f'    "Back"     -> Move Backward for {self.move_duration:.1f}s (Auto-Stop)')
        print(f'    "Step"     -> Step Forward for {self.step_duration:.1f}s (Auto-Stop)')
        print(f'    "Left"     -> Turn Left for {self.turn_duration:.1f}s (Auto-Stop)')
        print(f'    "Right"    -> Turn Right for {self.turn_duration:.1f}s (Auto-Stop)')
        print('    "Stop"     -> Immediate Stop')
        print('  Press Ctrl+C to quit at any time.')
        print('=' * 65 + '\n')

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=2000,
            device=self.device,
            dtype='int16',
            channels=1,
            callback=self._audio_callback,
        ):
            try:
                while self.running:
                    now = time.monotonic()

                    # 1. Check if the active timed movement has completed
                    if self.action_deadline is not None and now >= self.action_deadline:
                        log.info('Timed action "%s" completed -> Auto-STOP', self.action_desc)
                        self.current_cmd = 'S'
                        self.action_desc = 'IDLE'
                        self.action_deadline = None
                        self.sender.send('S')
                        self.last_sent_time = now
                        print('>>> AUTO-STOPPED: [S] STOP <<<')

                    # 2. Process incoming audio packets
                    try:
                        data = self.audio_queue.get(timeout=0.03)
                        if self.recognizer.AcceptWaveform(data):
                            res = json.loads(self.recognizer.Result())
                            text = res.get('text', '').strip()
                            if text and not self.utterance_consumed:
                                self._handle_phrase(text, is_partial=False)
                            self.utterance_consumed = False
                        else:
                            partial = json.loads(self.recognizer.PartialResult())
                            partial_text = partial.get('partial', '').strip()
                            if partial_text and not self.utterance_consumed:
                                self._handle_phrase(partial_text, is_partial=True)
                    except queue.Empty:
                        pass

                    # 3. Heartbeat watchdog while action is actively running
                    now = time.monotonic()
                    if (now - self.last_sent_time) >= WATCHDOG_INTERVAL_S:
                        self.sender.send(self.current_cmd)
                        self.last_sent_time = now

            except KeyboardInterrupt:
                log.info('Keyboard interrupt received.')
            finally:
                self.stop()

    def _handle_phrase(self, phrase: str, is_partial: bool = False):
        words = phrase.lower().split()
        tag = '[PARTIAL]' if is_partial else '[CONFIRMED]'

        for word in reversed(words):
            # Only match the exact allowed 6 commands
            if word == 'stop':
                self._trigger_stop(tag, word)
                self.utterance_consumed = True
                break

            elif word == 'forward':
                self._trigger_timed('F', self.move_duration, 'FORWARD', tag, word)
                self.utterance_consumed = True
                break

            elif word == 'back':
                self._trigger_timed('B', self.move_duration, 'BACKWARD', tag, word)
                self.utterance_consumed = True
                break

            elif word == 'step':
                self._trigger_timed('F', self.step_duration, 'STEP FORWARD', tag, word)
                self.utterance_consumed = True
                break

            elif word == 'left':
                self._trigger_timed('L', self.turn_duration, 'TURN LEFT', tag, word)
                self.utterance_consumed = True
                break

            elif word == 'right':
                self._trigger_timed('R', self.turn_duration, 'TURN RIGHT', tag, word)
                self.utterance_consumed = True
                break

    def _trigger_stop(self, tag: str, word: str):
        self.current_cmd = 'S'
        self.action_desc = 'IDLE'
        self.action_deadline = None
        self.sender.send('S')
        self.last_sent_time = time.monotonic()
        log.info('%s Heard "%s" -> STOP (S)', tag, word)
        print('>>> ACTIVE COMMAND: [S] STOP <<<')

    def _trigger_timed(self, cmd: str, duration: float, desc: str, tag: str, word: str):
        now = time.monotonic()
        self.current_cmd = cmd
        self.action_desc = desc
        self.action_deadline = now + duration
        self.sender.send(cmd)
        self.last_sent_time = now
        log.info('%s Heard "%s" -> %s (%s) for %.1fs', tag, word, desc, cmd, duration)
        print(f'>>> ACTIVE COMMAND: [{cmd}] {desc} (Duration: {duration:.1f}s) <<<')

    def stop(self):
        self.running = False
        self.current_cmd = 'S'
        self.action_deadline = None
        log.info('Stopping car and closing voice controller...')
        if self.sender:
            self.sender.send('S')
            time.sleep(0.1)
            self.sender.close()
        log.info('Done.')


# ===========================================================================
# ENTRY POINT
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='Voice-Controlled RC Car Controller with Timed Incremental Movements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  # Default (Mac mic + Auto BLE to BT24):\n'
            '  python voice_car.py\n'
            '\n'
            '  # Test speech recognition only (no car):\n'
            '  python voice_car.py --no-serial\n'
            '\n'
            '  # Adjust durations (e.g. forward/back 1.5s, step 0.3s, turn 0.4s):\n'
            '  python voice_car.py --move-duration 1.5 --step-duration 0.3 --turn-duration 0.4\n'
            '\n'
            '  # List available audio devices:\n'
            '  python voice_car.py --list-devices\n'
        ),
    )
    p.add_argument(
        '--ble', action='store_true', default=False,
        help='Connect directly to Bluetooth Low Energy (BLE) module (e.g. BT24 / HM-10).',
    )
    p.add_argument(
        '--ble-name', default='BT24',
        help='Target BLE device name (default: BT24).',
    )
    p.add_argument(
        '--port', default=None,
        help='Classic Bluetooth or USB COM port (e.g. COM5, /dev/cu.HC-06).',
    )
    p.add_argument(
        '--no-serial', action='store_true', default=False,
        help='Dry-run: recognize voice commands and log them but do not connect to Arduino.',
    )
    p.add_argument(
        '--move-duration', type=float, default=DEFAULT_MOVE_DURATION_S,
        help=f'Duration in seconds for "forward" and "back" before auto-stopping (default: {DEFAULT_MOVE_DURATION_S}s).',
    )
    p.add_argument(
        '--step-duration', type=float, default=DEFAULT_STEP_DURATION_S,
        help=f'Duration in seconds for "step" command before auto-stopping (default: {DEFAULT_STEP_DURATION_S}s).',
    )
    p.add_argument(
        '--turn-duration', type=float, default=DEFAULT_TURN_DURATION_S,
        help=f'Duration in seconds for "left" and "right" turns before auto-stopping (default: {DEFAULT_TURN_DURATION_S}s).',
    )
    p.add_argument(
        '--device', type=int, default=None,
        help='Audio input device index (run with --list-devices to view options).',
    )
    p.add_argument(
        '--list-devices', action='store_true', default=False,
        help='Print available microphone devices and exit.',
    )
    p.add_argument(
        '--lang', default='en-us',
        help='Vosk language model to use (default: en-us).',
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        print('Available Audio Devices:')
        print(sd.query_devices())
        return

    # Initialize communication
    if args.no_serial:
        sender = ArduinoSender(port=None, baud=SERIAL_BAUD, dry_run=True)
    elif args.ble or args.port is None or ('Bluetooth-Incoming-Port' in (args.port or '')):
        if args.port and 'Bluetooth-Incoming-Port' in args.port:
            log.warning("Notice: '%s' is a macOS internal port, not the car.", args.port)
            log.info("Auto-connecting to car via BLE (device: %s)...", args.ble_name)
        sender = ArduinoBleSender(target_name=args.ble_name)
    else:
        sender = ArduinoSender(port=args.port, baud=SERIAL_BAUD, dry_run=False)

    controller = VoiceCarController(
        sender=sender,
        device=args.device,
        lang=args.lang,
        move_duration=args.move_duration,
        step_duration=args.step_duration,
        turn_duration=args.turn_duration,
    )
    controller.run()


if __name__ == '__main__':
    main()
