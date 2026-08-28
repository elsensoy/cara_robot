#!/usr/bin/env python3
import rclpy
import random
import time
from rclpy.node import Node
from std_msgs.msg import String


class CaraBehaviorNode(Node):
    def __init__(self):
        super().__init__('cara_behavior_node')
        self.sub_emotion = self.create_subscription(
            String, '/cara/emotion_state', self.emotion_callback, 10)
        self.sub_mode = self.create_subscription(
            String, '/cara/mind_mode', self.mode_callback, 10)
        self.pub = self.create_publisher(
            String, '/cara/behavior_cmd', 10)

        self.latest_label = 'neutral'
        self.latest_conf = 0.0
        self.current_mode = 'BETA_REASONING'

        # Blink scheduling
        self.last_blink_time = 0.0
        self.next_blink_interval = self._sample_blink_interval()

        self.timer = self.create_timer(0.4, self.publish_behavior)

    def _sample_blink_interval(self):
        # Alpha mode (emotional support) → slower, softer blinks
        if self.current_mode == 'ALPHA_SUPPORTIVE':
            return random.uniform(1.5, 3.0)
        # Theta mode (creative) → quicker, playful blinks
        if self.current_mode == 'THETA_CREATIVE':
            return random.uniform(1.0, 2.5)
        # Beta / neutral → natural rate
        return random.uniform(3.0, 6.0)

    def mode_callback(self, msg):
        self.current_mode = msg.data.strip()

    def emotion_callback(self, msg):
        try:
            label, conf = msg.data.split(',')
            self.latest_label = label.strip().lower()
            self.latest_conf = float(conf.strip())
        except Exception as e:
            self.get_logger().warn(f'Bad emotion msg: {msg.data} ({e})')

    def should_blink_now(self):
        now = time.monotonic()
        if now - self.last_blink_time >= self.next_blink_interval:
            self.last_blink_time = now
            self.next_blink_interval = self._sample_blink_interval()
            return True
        return False

    def publish_behavior(self):
        # Neck pan/tilt is owned exclusively by the gaze tracker (model_gaze_mapper
        # → servo_pca9685_node). Behavior node only controls blink rate so the two
        # systems never fight over the same servos.
        blink = 1 if self.should_blink_now() else 0
        msg = String()
        msg.data = f'BLINK:{blink}'
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CaraBehaviorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
