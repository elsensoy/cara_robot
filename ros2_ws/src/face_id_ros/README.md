
### Install ROS2-humble

sudo apt update && sudo apt install -y curl gnupg2 lsb-release

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(lsb_release -sc) main" | \
sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop

### ros2 jazzy(for noble 24.04)
sudo apt update && sudo apt install -y curl gnupg2 lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

### add ROS 2 apt repo for *noble*
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(lsb_release -sc) main" | \
sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-jazzy-desktop
#### source it zsh
echo "source /opt/ros/jazzy/setup.zsh" >> ~/.zshrc
source /opt/ros/jazzy/setup.zsh
####  or bash 
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source /opt/ros/jazzy/setup.bash

#### instal ROS deps (Jazzy names)
sudo apt install -y \
  build-essential cmake git libopencv-dev \
  ros-jazzy-rclcpp \
  ros-jazzy-sensor-msgs \
  ros-jazzy-std-msgs \
  ros-jazzy-image-transport \
  ros-jazzy-cv-bridge \
  ros-jazzy-vision-msgs \
  ros-jazzy-rosidl-default-generators \
  ros-jazzy-v4l2-camera \
  ros-jazzy-rqt-image-view

### place models
models/yunet.onnx and models/arcface.onnx 

### build

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
./face_id

cd ~/elida/face-detect/face-rv     # or ~/cara/ws_face if that’s your workspace
rm -rf build install log
source /opt/ros/${ROS_DISTRO}/setup.zsh
colcon build --packages-select face_id_ros


## ROS2 Integration:

OpenCV (with dnn), cv_bridge, image_transport, vision_msgs

```
ws_face
├─ src/
│  └─ face_id_ros/       package folder (name must match <name> in package.xml)
│     ├─ package.xml
│     ├─ CMakeLists.txt
│     ├─ src/
│     │  └─ face_id_node.cpp        your C++ sources
│     ├─ include/
│     │  └─ face_id_ros/
│     │     ├─ detector.hpp
│     │     ├─ embedder.hpp
│     │     └─ face_db.hpp
│     ├─ srv/
│     │  └─ Enroll.srv
│     ├─ config/
│     │  └─ params.yaml
│     ├─ launch/
│     │  └─ face_id.launch.py
│     └─ models/
│        ├─ yunet.onnx
│        └─ arcface.onnx
├─ build/      created by colcon (don’t put this inside your package)
├─ install/    created by colcon
└─ log/       created by colcon

```


to make sure all is placed properly.

tree -L 3 ~/elida/face-detect/ws_face
sed -n '1,120p' ~/elida/face-detect/ws_face/src/face_id_ros/CMakeLists.txt
sed -n '1,120p' ~/elida/face-detect/ws_face/src/face_id_ros/package.xml

~/elida/face-detect/ws_face/
  └─ src/
      └─ face_id_ros/
### then build 
cd ~/elida/face-detect/ws_face
source /opt/ros/jazzy/setup.bash    
colcon build --packages-select face_id_ros
source install/setup.bash


# give your user access to /dev/video*
sudo usermod -a -G video $USER
# log out & back in (or reboot) so the group applies

# switch AE off, then set exposure (example values)
ros2 param set /v4l2_camera auto_exposure 1
ros2 param set /v4l2_camera exposure_time_absolute 200
### terminal A
source /opt/ros/jazzy/setup.bash
# pick one:
# A) lower-res fast debug:
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p image_size:="[640,480]" -p pixel_format:="MJPG" -p output_encoding:="rgb8"

# B) 720p:
# ros2 run v4l2_camera v4l2_camera_node --ros-args \
#   -p image_size:="[1280,720]" -p pixel_format:="MJPG" -p output_encoding:="rgb8"

### terminal B
source ~/elida/elida/face-detect/ws_face/install/setup.bash
ros2 launch face_id_ros face_id.launch.py

### Enroll once (face visible)
ros2 service call /face_id/enroll face_id_ros/srv/Enroll "{name: 'Alice'}"

#### fix YUYV --> RGB warning without MJPG (keep YUYV and match output):
```

  ros2 run v4l2_camera v4l2_camera_node --ros-args \
    -p image_size:="[640,480]" -p pixel_format:="YUYV" -p output_encoding:="yuv422_yuy2"
```
(Then convert to BGR in your node; cv_bridge can handle it.)

If a control says inactive (e.g., Exposure Time, Absolute): disable auto first,
then set the absolute value (as shown above). Some UVC cams won’t let you change it mid-stream.

Still seeing “permission denied” on controls? Some OEM drivers lock certain controls. also test with:
```
  sudo apt install v4l-utils
  v4l2-ctl -d /dev/video0 --all
  v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=1 --set-ctrl=exposure_absolute=200
```

(If that fails as non-root but works as root, it’s a permissions thing—ensure you’re in the video group and re-login.)

---
detector.hpp/.cpp: wraps OpenCV DNN with a lightweight face detector (e.g., YuNet ONNX). outputs std::vector<FaceBox>{cv::Rect box; float score;}.

embedder.hpp/.cpp: wraps ArcFace ONNX to produce a 512-D, L2-normalized embedding.

face_db.hpp/.cpp: in-memory list of (name, embedding); cosine match; YAML save/load.

#### Use a Logitech webcam ( /dev/video0)
ros2 run v4l2_camera v4l2_camera_node
#### It typically publishes /image_raw or /camera/image

If the topic is not /camera/image, either:

-    edit camera_topic in config/params.yaml, or

-    remap in the launch file (see below).

### launch
```
    ros2 launch face_id_ros face_id.launch.py
```

### enroll face once
```

ros2 service call /face_id/enroll face_id_ros/srv/Enroll "{name: 'Elida'}"

```
this stores embedding to db_path (/tmp/known_faces.yaml by default).

### view outputs
rqt_image_view  # choose /face_id/annotated
ros2 topic echo /face_id/detections #structured
ros2 topic echo /face_id/recognized #simple recognized names

### Quick troubleshooting

No camera frames? Make sure v4l2_camera is running and the topic name matches camera_topic.

Unsatisfied dependencies at build: re-check THE installed vision-msgs, cv-bridge, image-transport, and sourced ROS (setup.bash) before colcon build.

Low FPS on CPU: set use_gpu: false (already default), reduce input resolution (e.g., run camera at 640×480), or later enable GPU by building OpenCV with CUDA / using ONNX Runtime CUDA.

Mislabels / “Unknown”: lower recog_thresh to ~0.35, and re-enroll capturing a few seconds of your face (implementation can average internally).

Wrong model paths: double-check package://face_id_ros/models/... paths or swap to absolute paths.

--- 

For better accuracy, add face alignment (5-landmark similarity transform to 112×112) before embedding; ArcFace expects aligned crops.

For stability, keep a rolling median of embeddings during enrollment (e.g., 10 frames) and save the mean.

If you need ROS-native face databases, swap the YAML for a ros2 param or a small face_id_db node that serves queries over a service.


### deps
```
    sudo apt update
    sudo apt install -y \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-rqt-image-view \
    ros-${ROS_DISTRO}-v4l2-camera \
    build-essential cmake git
```
sudo apt update
sudo apt install ros-humble-cv-bridge


### ENABLE GPU 
batching is overkill for a webcam, but pre-allocate Mats and reuse blobs to avoid churn.

Enable GPU: Build OpenCV with CUDA or use ONNXRuntime/TensorRT backends for 2–5× speed on NVIDIA GPUs.

Face alignment: Slight boost—similarity transform using detected landmarks before embedding (ArcFace expects aligned 112×112).

Multi-threading: One capture thread -> lock-free queue -> worker thread (detect+embed) -> UI thread (draw). Keeps UI smooth.


## things to consider 

Input size & preprocessing: YuNet often expects a specific input size (e.g., 320×320 or similar) and then you set the input size in code (see OpenCV tutorial). 
OpenCV
+1

Model version / accuracy trade-offs: There are “lightweight” vs “heavy” versions. For example the ArcFace ResNet100 version is large (~166 MB) for high accuracy. 
Yakhyo’s Blog
+1

Licensing & usage: Make sure you check the model license in the repo you download from. Many are under MIT / Apache but verify for commercial use if needed.

Model names: If you rename the file (e.g., arcface.onnx), ensure your code’s path matches and that the model’s input/output expectations align with your embedder wrapper.

ONNX compatibility: Some ONNX models may require a compatible version of OpenCV DNN or ONNX Runtime. If you get errors “unsupported format”, you may need a newer OpenCV or convert the model. 
NVIDIA Developer Forums


## handy reset 
cd /home/elida/cara/ws_face
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --packages-select face_id_ros
source install/setup.bash