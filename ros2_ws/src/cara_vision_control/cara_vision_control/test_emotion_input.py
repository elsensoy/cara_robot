#!/usr/bin/env python3
"""
Manual emotion injector for testing CARA's behavior pipeline without a camera.

Publishes fake emotion messages to /cara/emotion_state in the format:
    "<label>,<confidence>"
e.g. "sad,0.85"

Usage:
  Run alongside cara_behavior_node and cara_arduino_bridge_node.
  Type emotions interactively, or use --auto to cycle through them.
"""
import argparse
import sys
import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


VALID_EMOTIONS = ['neutral', 'happy', 'sad', 'surprise', 'angry', 'fear', 'disgust']


class EmotionInjector(Node):
    def __init__(self):
        super().__init__('emotion_injector')
        self.pub = self.create_publisher(String, '/cara/emotion_state', 10)
        self.get_logger().info('Emotion injector ready. Publishing to /cara/emotion_state')

    def publish(self, label, confidence):
        msg = String()
        msg.data = f'{label},{confidence:.2f}'
        self.pub.publish(msg)
        self.get_logger().info(f'Sent: {msg.data}')


# -----------------------------------------------------------------
# --- Interactive mode: type emotions at the prompt -----------------
# -----------------------------------------------------------------

def interactive_loop(node):
    print('\n=== Interactive Emotion Test ===')
    print(f'Valid labels: {", ".join(VALID_EMOTIONS)}')
    print('Format: <label> [confidence]   (confidence defaults to 0.9)')
    print('Special commands: quit, hold <label> <seconds>')
    print('Example:  sad 0.85')
    print('Example:  hold happy 5')
    print('--------------------------------\n')

    while rclpy.ok():
        try:
            raw = input('emotion> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue
        if raw in ('quit', 'exit', 'q'):
            break

        parts = raw.split()

        # "hold <label> <seconds>" — publish continuously for N seconds
        if parts[0] == 'hold' and len(parts) >= 3:
            label = parts[1]
            try:
                duration = float(parts[2])
            except ValueError:
                print('  Bad duration')
                continue
            conf = float(parts[3]) if len(parts) > 3 else 0.9
            _hold(node, label, conf, duration)
            continue

        # "<label> [confidence]"
        label = parts[0]
        if label not in VALID_EMOTIONS:
            print(f'  Unknown emotion. Pick from: {", ".join(VALID_EMOTIONS)}')
            continue
        try:
            conf = float(parts[1]) if len(parts) > 1 else 0.9
        except ValueError:
            print('  Bad confidence value')
            continue
        if not 0.0 <= conf <= 1.0:
            print('  Confidence must be 0.0–1.0')
            continue

        # Single publish — but behavior node smooths over time, so we
        # also hold it briefly to give the head time to move.
        _hold(node, label, conf, 1.0)


def _hold(node, label, confidence, duration):
    """Republish the emotion at 5Hz for `duration` seconds."""
    end = time.monotonic() + duration
    print(f'  Holding "{label}" @ {confidence:.2f} for {duration:.1f}s...')
    while time.monotonic() < end and rclpy.ok():
        node.publish(label, confidence)
        time.sleep(0.2)


# -----------------------------------------------------------------
# --- Auto mode: cycle through every emotion -----------------------
# -----------------------------------------------------------------

def auto_loop(node, hold_seconds=4.0):
    """Cycle through each emotion automatically, useful for headless tests."""
    sequence = [
        ('neutral', 0.9),
        ('happy', 0.9),
        ('sad', 0.85),     # should trigger faster blinks
        ('sad', 0.95),     # high confidence sad
        ('surprise', 0.9),
        ('happy', 0.4),    # low confidence -> should ignore (return to 90)
        ('neutral', 0.9),
    ]
    print(f'\n=== Auto Mode — {hold_seconds}s per emotion ===\n')
    for label, conf in sequence:
        if not rclpy.ok():
            break
        _hold(node, label, conf, hold_seconds)
        time.sleep(0.5)  # short gap between transitions
    print('\nAuto sequence complete.')


# -----------------------------------------------------------------
# --- Main ---------------------------------------------------------
# -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Test CARA emotion behavior without camera')
    parser.add_argument('--auto', action='store_true',
                        help='Cycle through all emotions automatically')
    parser.add_argument('--hold', type=float, default=4.0,
                        help='Seconds to hold each emotion in auto mode (default: 4)')
    args = parser.parse_args()

    rclpy.init()
    node = EmotionInjector()

    # Spin in background so the node stays alive for callbacks/logging
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        if args.auto:
            auto_loop(node, hold_seconds=args.hold)
        else:
            interactive_loop(node)
    except KeyboardInterrupt:
        pass
    finally:
        print('\nShutting down injector.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
