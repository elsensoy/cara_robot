#include <opencv2/opencv.hpp>
#include "detector.hpp"
#include "embedder.hpp"
#include "face_db.hpp"

int main(){
  Detector det("models/yunet.onnx", 0.6f);
  Embedder emb("models/arcface.onnx");
  FaceDB db; db.load("data/known_faces.yaml");

  cv::VideoCapture cap(0);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

  bool enroll=false; std::string enroll_name="Alice";
  std::cout<<"Keys: [E]nroll toggle, [S]ave DB, [ESC] quit\n";

  while(true){
    cv::Mat frame; if(!cap.read(frame)) break;
    auto faces = det.detect(frame);

    for(const auto& f: faces){
      cv::Rect r = f.box & cv::Rect(0,0,frame.cols,frame.rows);
      cv::Mat face = frame(r).clone();
      cv::Mat vec = emb.embed(face); // 1x512, L2 normed

      std::string label="Unknown";
      if(!enroll){
        auto hit = db.best(vec);  // cosine similarity in [-1,1]
        if(hit.score > 0.40f) label = hit.name; // tune threshold
      } else {
        label = "ENROLL: "+enroll_name;
        db.add(enroll_name, vec); // collect a few frames; press S to save
      }

      cv::rectangle(frame, r, {0,255,0}, 2);
      cv::putText(frame, label, {r.x, std::max(0, r.y-8)}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,255,0}, 2);
    }

    cv::imshow("FaceID C++", frame);
    int k = cv::waitKey(1) & 0xFF;
    if(k==27) break;
    if(k=='e' || k=='E') enroll = !enroll;
    if(k=='s' || k=='S') db.save("data/known_faces.yaml");
  }
}

