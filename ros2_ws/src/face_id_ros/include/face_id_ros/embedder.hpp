// embedder.hpp
#pragma once
#include <opencv2/opencv.hpp>
class Embedder {
public:
  explicit Embedder(const std::string& onnx_path);
  cv::Mat embed(const cv::Mat& face_bgr); // returns 1x512 float (L2-normalized)
private:
  cv::dnn::Net net_;
};
