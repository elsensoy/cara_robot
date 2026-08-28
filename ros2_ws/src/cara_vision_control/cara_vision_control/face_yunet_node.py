#!/usr/bin/env python3
import os
import cv2 as cv
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

class FaceYuNetNode(Node):
    """
    ROS2 wrapper around YuNet face detector.
    
    Subscribes:
      - /image_raw (sensor_msgs/Image, bgr8)
    
    Publishes:
      - /faces/primary_center (geometry_msgs/Point)
      - /faces/debug_image (sensor_msgs/Image)
      - /cara/face_crop (sensor_msgs/Image) -> NEW: For Emotion Node
    """

    def __init__(self):
        super().__init__("face_yunet_node")

        self.bridge = CvBridge()

        # Declare params
        self.declare_parameter("camera_topic", "/image_raw")
        self.declare_parameter("detector_model", "")
        self.declare_parameter("score_threshold", 0.6)
        self.declare_parameter("nms_threshold", 0.3)
        self.declare_parameter("top_k", 5000)
        self.declare_parameter("show_window", True)

        camera_topic = self.get_parameter("camera_topic").value
        model_path = self.get_parameter("detector_model").value
        score_th = float(self.get_parameter("score_threshold").value)
        nms_th = float(self.get_parameter("nms_threshold").value)
        top_k = int(self.get_parameter("top_k").value)
        self.show_window = self.get_parameter("show_window").value

        # Default model path
        if not model_path:
            share = get_package_share_directory("cara_vision_control")
            model_path = os.path.join(share, "models", "yunet.onnx")

        self.get_logger().info(f"Loading YuNet model from: {model_path}")
        self.det = cv.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=score_th,
            nms_threshold=nms_th,
            top_k=top_k,
        )

        # ROS I/O
        self.sub = self.create_subscription(
            Image, camera_topic, self.image_callback, 10
        )
        self.pub_center = self.create_publisher(Point, "/faces/primary_center", 10)
        self.pub_debug = self.create_publisher(Image, "/faces/debug_image", 10)
        
        # --- NEW PUBLISHER FOR EMOTION NODE ---
        self.pub_face_crop = self.create_publisher(Image, "/cara/face_crop", 10)

        self.get_logger().info(f"FaceYuNetNode started. watching {camera_topic}")

    def image_callback(self, msg: Image):
        # 1. Convert to OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        h, w, _ = frame.shape
        self.det.setInputSize((w, h))

        # 2. Inference
        _, faces = self.det.detect(frame)
        found_faces = faces if faces is not None else []

        # 3. Visualization & Logic Loop
        if len(found_faces) > 0:
            #  VISUALIZATION LOOP  
            for face in found_faces:
                coords = face[:-1].astype(np.int32)
                cv.rectangle(frame, (coords[0], coords[1]), 
                             (coords[0]+coords[2], coords[1]+coords[3]), 
                             (0, 255, 0), 2)
                conf = face[-1]
                cv.putText(frame, f"{conf:.2f}", (coords[0], coords[1]-10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            #  CONTROL LOGIC (BEST FACE)  
            best_idx = np.argmax(found_faces[:, -1])
            best = found_faces[best_idx]
            x, y, w_box, h_box, *rest = best
            score = best[-1]

            cx = x + w_box * 0.5
            cy = y + h_box * 0.5

            # Publish Center Point
            pt = Point()
            pt.x = float(cx)
            pt.y = float(cy)
            pt.z = float(score)
            self.pub_center.publish(pt)

            # NEW: Extract and publish face crop for emotion node  
            # Ensure integers for slicing
            x_int, y_int, w_int, h_int = int(x), int(y), int(w_box), int(h_box)

            # Add a little margin (20%)
            margin = int(0.2 * w_int)
            x1 = max(0, x_int - margin)
            y1 = max(0, y_int - margin)
            x2 = min(w, x_int + w_int + margin)
            y2 = min(h, y_int + h_int + margin)

            # Crop from the frame
            face_crop = frame[y1:y2, x1:x2]

            # Publish if valid
            if face_crop.size > 0:
                try:
                    crop_msg = self.bridge.cv2_to_imgmsg(face_crop, encoding="bgr8")
                    self.pub_face_crop.publish(crop_msg)
                except Exception as e:
                    self.get_logger().warn(f"Failed to publish crop: {e}")

        # 4. Display Window
        if self.show_window:
            cv.imshow("Cara Face Debug", frame)
            cv.waitKey(1)

        # 5. Publish Debug Image
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.pub_debug.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FaceYuNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
