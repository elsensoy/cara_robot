#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point.hpp>      // or your custom detection msg
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "face_id_ros/detector.hpp"
#include "face_id_ros/embedder.hpp"
#include "face_id_ros/face_db.hpp"

class FaceIdNode : public rclcpp::Node
{
public:
  FaceIdNode()
  : Node("face_id_node")
  {
    // 1) Declare parameters
    camera_topic_ = this->declare_parameter<std::string>("camera_topic", "/image_raw");
    std::string detector_model = this->declare_parameter<std::string>("detector_model", "models/yunet.onnx");
    std::string embedder_model = this->declare_parameter<std::string>("embedder_model", "models/arcface.onnx");
    std::string db_path = this->declare_parameter<std::string>("db_path", "/tmp/known_faces.yaml");
    det_thresh_ = this->declare_parameter<double>("det_thresh", 0.6);
    recog_thresh_ = this->declare_parameter<double>("recog_thresh", 0.4);

    // 2) Init detector / embedder / DB
    det_ = std::make_unique<Detector>(detector_model, static_cast<float>(det_thresh_));
    emb_ = std::make_unique<Embedder>(embedder_model);
    db_.load(db_path);

    // 3) Publisher for detections
    // TODO: replace geometry_msgs::msg::Point with  s actual detection message type
    // det_pub_ = this->create_publisher<geometry_msgs::msg::Point>("/faces/primary_center", 10);
    det_pub_ = this->create_publisher<geometry_msgs::msg::Point>("/faces/primary_center", 10);

    // 4) Subscription to camera images
    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      camera_topic_, rclcpp::SensorDataQoS(),
      std::bind(&FaceIdNode::image_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "FaceIdNode started, subscribing to %s", camera_topic_.c_str());
  }

private:
void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  cv::Mat frame;
  try {
    frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
  } catch (const std::exception &e) {
    RCLCPP_WARN(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  RCLCPP_INFO(this->get_logger(),
              "image_callback: %dx%d", frame.cols, frame.rows);

auto faces = det_->detect(frame);
RCLCPP_INFO(this->get_logger(),
            "detected %zu faces", faces.size());

if (faces.empty()) {
  return;
}

// choose best by score
const auto &f = faces[0];
const cv::Rect &r = f.box;

geometry_msgs::msg::Point pt;
pt.x = static_cast<double>(r.x + r.width * 0.5);
pt.y = static_cast<double>(r.y + r.height * 0.5);
pt.z = f.score;

RCLCPP_INFO(this->get_logger(),
            "publishing center: (%.1f, %.1f) score=%.2f",
            pt.x, pt.y, pt.z);

det_pub_->publish(pt);


  std::string camera_topic_;
  double det_thresh_;
  double recog_thresh_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr det_pub_;

  std::unique_ptr<Detector> det_;
  std::unique_ptr<Embedder> emb_;
  FaceDB db_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FaceIdNode>());
  rclcpp::shutdown();
  return 0;
}

