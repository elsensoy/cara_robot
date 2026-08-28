
#include "face_id_ros/embedder.hpp"
using namespace cv;
Embedder::Embedder(const std::string& path){
  net_ = dnn::readNetFromONNX(path);
}
cv::Mat Embedder::embed(const Mat& face_bgr){
  Mat rgb; cvtColor(face_bgr, rgb, COLOR_BGR2RGB);
  Mat resized; resize(rgb, resized, Size(112,112));
  Mat blob = dnn::blobFromImage(resized, 1.0/128.0, Size(112,112), Scalar(127.5,127.5,127.5), true, false);
  net_.setInput(blob);
  Mat out = net_.forward();                // [1 x 512]
  // L2 normalize
  Mat e; normalize(out, e, 1.0, 0.0, NORM_L2);
  return e;                                // CV_32F 1x512
}
