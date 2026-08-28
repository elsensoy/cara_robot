#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import torch
import cv2
from collections import deque  # <--- 1. NEW IMPORT

# Ensure these match your actual package structure
from cara_vision_control.cara_emotion_core import PersonalizedEmotionViT, InteractiveLearningSystem

class CaraEmotionNode(Node):
    def __init__(self):
        super().__init__('cara_emotion_node')
        self.bridge = CvBridge()
        
        # --- 1. Load Model & System ---
        self.get_logger().info("Loading ViT Model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Model
        self.model = PersonalizedEmotionViT().to(self.device)
        self.model.eval() 
        
        # Initialize Learning System
        self.learning_system = InteractiveLearningSystem(self.model)
        self.get_logger().info(f"Model loaded on {self.device}")

        #  2. Subscribers  
        # Use qos_profile_sensor_data instead of 10
        self.sub = self.create_subscription(Image, '/cara/face_crop', self.process_face, qos_profile_sensor_data)
        self.sub_feedback = self.create_subscription(String, '/cara/feedback', self.handle_feedback, 10)
        self.sub_train = self.create_subscription(Bool, '/cara/train', self.handle_train, 10)

        #  3. publishers  
        self.pub_emotion = self.create_publisher(String, '/cara/emotion', 10)
        self.pub_emotion_state = self.create_publisher(String, '/cara/emotion_state', 10)
        self.current_frame = None
        self.is_training = False

        #  4. MEMORY BUFFER (NEW) 
        # Stores the last 30 face crops (approx 2-3 seconds depending on frame rate)
        self.face_buffer = deque(maxlen=30) 

    def process_face(self, msg):
        # Skip processing if we are currently training
        if self.is_training:
            return

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_frame = frame_bgr

            # --- NEW: Add to Short-Term Memory ---
            # We save a COPY so it doesn't get overwritten
            self.face_buffer.append(frame_bgr.copy()) 

            # Run Prediction
            result = self.model.predict_frame(frame_bgr)
            
            if result is None:
                self.get_logger().info("Model returned None (Skipping frame)")
                return

            label = result['primary_emotion'].strip().lower()
            confidence = float(result['confidence'])

            # human-readable
            emotion_str = f"{label} ({confidence:.2f})"
            self.pub_emotion.publish(String(data=emotion_str))

            # machine-friendly
            emotion_state_str = f"{label},{confidence:.2f}"
            self.pub_emotion_state.publish(String(data=emotion_state_str))

        except Exception as e:
            self.get_logger().warn(f"Inference error: {e}")

    def handle_feedback(self, msg):
        """
        Receives corrected labels (e.g. 'happy').
        Saves the entire buffer (last 3 seconds) to the dataset.
        """
        label = msg.data.split(':')[-1].strip().lower() # handle "SAVE:happy" or just "happy"
        
        if label not in self.model.emotion_names:
            self.get_logger().warn(f"Invalid label received: {label}")
            return

        if len(self.face_buffer) > 0:
            self.get_logger().info(f"--- LEARNING TRIGGERED: Saving {len(self.face_buffer)} frames as '{label}' ---")
            
            # Loop through memory and save every face crop we saw recently
            count = 0
            for frame in list(self.face_buffer):
                self.learning_system.save_labeled_sample(frame, label)
                count += 1
            
            self.get_logger().info(f"Successfully saved {count} samples.")
            
            # Optional: Clear buffer so we don't save the same faces twice
            self.face_buffer.clear()
        else:
            self.get_logger().warn("Cannot save: Memory buffer is empty.")

    def handle_train(self, msg):
        """
        Triggered when /cara/train receives True.
        """
        if msg.data: 
            self.is_training = True
            self.get_logger().info("--- STARTING TRAINING (Feed Paused) ---")
            
            try:
                success, val_acc = self.learning_system.update_model(
                    epochs=10, 
                    batch_size=4, 
                    val_ratio=0.2
                )
                
                if success:
                    metric_str = f"{val_acc:.3f}" if val_acc is not None else "N/A"
                    self.get_logger().info(f"--- TRAINING COMPLETE | Best Val Acc: {metric_str} ---")
                    
                    current_temp = self.model.temperature.item()
                    self.get_logger().info(f"--- NEW TEMPERATURE: {current_temp:.3f} ---")
                else:
                    self.get_logger().warn("Training skipped (insufficient data < 8 samples)")
            
            except Exception as e:
                self.get_logger().error(f"Training failed: {e}")
            
            finally:
                self.is_training = False
                self.get_logger().info("--- RESUMING INFERENCE ---")

def main(args=None):
    rclpy.init(args=args)
    node = CaraEmotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
