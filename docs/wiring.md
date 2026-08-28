1. The ROS 2 Architecture

To mimic the Disney Olaf setup, we’ll use a "Sense-Think-Act" pipeline. Each part is a separate ROS 2 Node communicating over a shared "Brain" topic.

    imu_node: Reads orientation from the BNO055 and publishes sensor_msgs/Imu.

    policy_node: Subscribes to IMU and Joint States, runs the ONNX "Brain," and publishes JointTrajectory.

    actuator_node: Subscribes to the trajectory and commands the two PCA9685 boards via I2C.
    2. The Integrated ROS 2 Controller Node

This Python script combines IMU reading and ONNX policy into a single ROS 2 Node. This is the code that will live on the Orin Nano.
```
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from trajectory_msgs.msg import JointTrajectory
import onnxruntime as ort
import numpy as np

class CaraBrainNode(Node):
    def __init__(self):
        super().__init__('cara_brain')
        
        # 1. Load  Disney-style RL Policy
        self.ort_session = ort.InferenceSession("cara_policy.onnx")
        
        # 2. Subscribe to the IMU (Balance Feedback)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        
        # 3. Publish to the Actuators (The 20 Servos)
        self.joint_pub = self.create_publisher(JointTrajectory, '/joint_commands', 10)
        
        self.get_logger().info("Cara's ROS 2 Brain is Online.")

    def imu_callback(self, msg):
        # Flatten IMU data into a policy observation vector
        # [Roll, Pitch, Yaw, Accel_X, Accel_Y, Accel_Z]
        obs = np.array([
            msg.orientation.x, msg.orientation.y, msg.orientation.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        ]).astype(np.float32).reshape(1, -1)
        
        # Run Inference (The "Think" step)
        action = self.ort_session.run(None, {"obs": obs})[0]
        
        # Publish the new joint targets
        self.send_servo_commands(action)

    def send_servo_commands(self, action):
        cmd = JointTrajectory()
        # Mapping 20 joints here...
        # ... logic to fill and publish JointTrajectory message
        self.joint_pub.publish(cmd)

def main():
    rclpy.init()
    node = CaraBrainNode()
    rclpy.spin(node)
    rclpy.shutdown() 
```
### Revised Wiring Strategy
Integrating A research-grade 20-servo body with ROS 2 transforms Cara from a collection of parts into a unified system. By using the Jetson Orin Nano Super, we got enough compute to run RL (Reinforcement Learning) policy in real-time while processing IMU data at 50Hz–100Hz.
#### 1. The ROS 2 Architecture

To mimic the Disney Olaf setup, we’ll use a "Sense-Think-Act" pipeline. Each part is a separate ROS 2 Node communicating over a shared "Brain" topic.

    imu_node: Reads orientation from the BNO055 and publishes sensor_msgs/Imu.

    policy_node: Subscribes to IMU and Joint States, runs the ONNX "Brain," and publishes JointTrajectory.

    actuator_node: Subscribes to the trajectory and commands the two PCA9685 boards via I2C.

#### 2. The Integrated ROS 2 Controller Node

This Python script combines  IMU reading and  ONNX policy into a single ROS 2 Node. This is the code that will live on the Orin Nano.
```
Python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from trajectory_msgs.msg import JointTrajectory
import onnxruntime as ort
import numpy as np

class CaraBrainNode(Node):
    def __init__(self):
        super().__init__('cara_brain')
        
        # 1. Load Disney-style RL Policy
        self.ort_session = ort.InferenceSession("cara_policy.onnx")
        
        # 2. Subscribe to the IMU (Balance Feedback)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        
        # 3. Publish to the Actuators (The 20 Servos)
        self.joint_pub = self.create_publisher(JointTrajectory, '/joint_commands', 10)
        
        self.get_logger().info("Cara's ROS 2 Brain is Online.")

    def imu_callback(self, msg):
        # Flatten IMU data into a policy observation vector
        # [Roll, Pitch, Yaw, Accel_X, Accel_Y, Accel_Z]
        obs = np.array([
            msg.orientation.x, msg.orientation.y, msg.orientation.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        ]).astype(np.float32).reshape(1, -1)
        
        # Run Inference (The "Think" step)
        action = self.ort_session.run(None, {"obs": obs})[0]
        
        # Publish the new joint targets
        self.send_servo_commands(action)

    def send_servo_commands(self, action):
        cmd = JointTrajectory()
        # Mapping 20 joints here...
        # ... logic to fill and publish JointTrajectory message
        self.joint_pub.publish(cmd)

def main():
    rclpy.init()
    node = CaraBrainNode()
    rclpy.spin(node)
    rclpy.shutdown()
```
#### 3. Setting Up the I2C "Daisy Chain"

Since the IMU (BNO055) and the Servo Boards (PCA9685) all use I2C, we simply connect them in parallel.

        Jetson Pin 3 (SDA) & Pin 5 (SCL) connect to the SDA/SCL pins of all three boards.

        Addressing is Key: * BNO055: 0x28

        PCA9685 Board 1: 0x40

        PCA9685 Board 2: 0x41 (Solder the A0 bridge)

#### 4. Sim-to-Real Synchronization

To ensure Cara’s real-world walk matches the Isaac Sim training:

    Update Rate: Ensure  ROS 2 node runs at the exact same frequency as simulation (usually 50Hz).

    Latency: Use the RMW_IMPLEMENTATION=rmw_cyclonedds_cpp environment variable on Jetson to minimize the delay between "feeling" a fall and "correcting" it with the servos.

#### 5. Final Safety Check

Since we are using  lab's 3D printer and the 5060 GPU for training, we have the high-end tools to succeed. Before powering the 20 servos:

    E-Stop: Always have a physical way to cut power to the servo battery. If the RL policy "glitches," 20 high-torque servos can easily crush 3D-printed parts or fingers.

    Homeostasis: In imu_callback, add a check for msg.linear_acceleration. If it detects a "freefall" (0g), have Cara go into a "tuck" position to protect her joints and the Orin Nano.
 
 
 For 20 servos on the Jetson, we need to scale up:

Component	Power Source	Logic Connection
Jetson Orin Nano Super	Dedicated 9V-20V (via DC Jack)	Common Ground with everything
PCA9685 (x2)	5V Logic from Jetson	I2C (SDA/SCL) to Jetson
20 Servos	High-Current 5V/6V BEC	PWM from PCA9685


[!IMPORTANT] The Diode/Transistor Tip: For  blinking/head servos, the transistor setup is fine for small loads, but for the 20-servo body, we will use the PCA9685 boards directly. We use a Common Ground—connect a GND pin from the Jetson to the GND on servo power supply. Without this, the PWM signals will be "floating" and servos will jitter violently.

