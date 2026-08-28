#pragma once
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <string>

struct DetectedFace {
  cv::Rect box;
  float score;
};

class Detector {
public:
  Detector(const std::string &model_path, float score_thresh);

  std::vector<DetectedFace> detect(const cv::Mat &bgr);

private:
  cv::Ptr<cv::FaceDetectorYN> yunet_;
  float score_thresh_;
};

