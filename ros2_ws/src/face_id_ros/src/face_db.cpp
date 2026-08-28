#include "face_id_ros/face_db.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>

using cv::Mat;
using std::string;

namespace {
float cosine(const Mat& a, const Mat& b) {
  // assumes L2-normalized rows; cosine = dot product
  return a.dot(b);
}
}  // namespace

void FaceDB::add(const std::string& n, const cv::Mat& e) {
  names_.push_back(n);
  embs_.push_back(e.clone());
}

Hit FaceDB::best(const cv::Mat& e) const {
  Hit h;
  h.name = "";
  h.score = -1.0f;

  if (embs_.empty()) {
    return h;
  }

  for (size_t i = 0; i < embs_.size(); ++i) {
    float s = cosine(embs_[i], e);
    if (s > h.score) {
      h.score = s;
      h.name = names_[i];
    }
  }
  return h;
}

bool FaceDB::save(const std::string& path) const {
  cv::FileStorage fs(path, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    return false;
  }

  fs << "names" << "[";
  for (const auto& n : names_) {
    fs << n;
  }
  fs << "]";

  fs << "embs" << "[";
  for (const auto& e : embs_) {
    fs << e;
  }
  fs << "]";

  fs.release();
  return true;
}

bool FaceDB::load(const std::string& path) {
  names_.clear();
  embs_.clear();

  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }

  cv::FileNode names_node = fs["names"];
  cv::FileNode embs_node  = fs["embs"];

  if (names_node.type() != cv::FileNode::SEQ ||
      embs_node.type()  != cv::FileNode::SEQ) {
    return false;
  }

  for (auto it = names_node.begin(); it != names_node.end(); ++it) {
    std::string n;
    (*it) >> n;
    names_.push_back(n);
  }

  for (auto it = embs_node.begin(); it != embs_node.end(); ++it) {
    cv::Mat e;
    (*it) >> e;
    embs_.push_back(e);
  }

  fs.release();
  return true;
}
