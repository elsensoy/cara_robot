#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

struct Hit {
  std::string name;
  float score;
};

class FaceDB {
public:
  // Add one embedding with a name
  void add(const std::string& n, const cv::Mat& e);

  // Return best match (by cosine similarity); if none, score = -1, name=""
  Hit best(const cv::Mat& e) const;

  // Save / load DB to/from a YAML file
  bool save(const std::string& path) const;
  bool load(const std::string& path);

private:
  std::vector<std::string> names_;
  std::vector<cv::Mat> embs_;  // each row: 1x512 CV_32F, L2-normalized
};
