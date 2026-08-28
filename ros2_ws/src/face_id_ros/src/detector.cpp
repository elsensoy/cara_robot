#include "face_id_ros/detector.hpp"
#include <opencv2/imgproc.hpp>
#include <stdexcept>

Detector::Detector(const std::string &model_path, float score_thresh)
  : score_thresh_(score_thresh)
{
  yunet_ = cv::FaceDetectorYN::create(
      model_path,
      "",
      cv::Size(320, 320),
      score_thresh_,
      0.3f,   // NMS
      5000
  );

  if (yunet_.empty()) {
    throw std::runtime_error("Failed to create FaceDetectorYN from: " + model_path);
  }
}

std::vector<DetectedFace> Detector::detect(const cv::Mat &bgr)
{
  std::vector<DetectedFace> out;
  if (bgr.empty()) return out;

  // Original size (for mapping back)
  int orig_w = bgr.cols;
  int orig_h = bgr.rows;

  // Resize to YuNet input size
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(320, 320));
  yunet_->setInputSize(cv::Size(320, 320));

  cv::Mat faces;
  yunet_->detect(resized, faces);

  if (faces.empty()) {
    return out;
  }

  // Scale factors back to original resolution
  float scale_x = static_cast<float>(orig_w) / 320.0f;
  float scale_y = static_cast<float>(orig_h) / 320.0f;

  for (int i = 0; i < faces.rows; ++i) {
    float x = faces.at<float>(i, 0);
    float y = faces.at<float>(i, 1);
    float w  = faces.at<float>(i, 2);
    float h  = faces.at<float>(i, 3);
    float score = faces.at<float>(i, 4);

    if (score < score_thresh_) continue;

    // Map back to original coordinates
    int x0 = static_cast<int>(x * scale_x);
    int y0 = static_cast<int>(y * scale_y);
    int ww = static_cast<int>(w * scale_x);
    int hh = static_cast<int>(h * scale_y);

    cv::Rect rect(x0, y0, ww, hh);
    rect &= cv::Rect(0, 0, orig_w, orig_h);
    if (rect.area() <= 0) continue;

    out.push_back({rect, score});
  }

  return out;
}

