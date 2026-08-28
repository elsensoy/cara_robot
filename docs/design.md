The new topic flow:
cara_main.py  →  /cara/mind_mode  →  behavior_node
                                       ├── head angle (mode-aware)
                                       └── blink rate (mode-aware)
                                             ↓
                                       /cara/behavior_cmd → hardware
                                       cd /workspace/ros2_ws
After rebuilding (colcon build --packages-select cara_vision_control from the workspace), the behavior node will need to be restarted to pick up the changes.
 
 
 Run these inside the container (docker compose exec cara_runtime bash):

# 1. Source the workspace
source /workspace/install/setup.bash

# 2. Start behavior_node in the background
ros2 run cara_vision_control behavior_node &

# 3. Confirm it has the new subscription
ros2 node info /cara_behavior_node

You should see /cara/mind_mode listed under Subscribers. Then:

# 4. Publish ALPHA mode and check the output
ros2 topic pub --once /cara/mind_mode std_msgs/msg/String '{data: "ALPHA_SUPPORTIVE"}'
ros2 topic echo /cara/be
# expect: HEAD:78,BLINK:timer)

# 5. Switch to THETA and
ros2 topic pub --once /ctring '{data:
"THETA_CREATIVE"}'
ros2 topic echo /cara/be
# expect: HEAD:100,...

# 6. Back to BETA
ros2 topic pub --once /cara/mind_mode std_msgs/msg/String '{data:"BETA_REASONING"}'
ros2 topic echo /cara/behavior_cmd --once
# expect: HEAD:90,...

The key things to verify:
- ros2 node info shows /cara/mind_mode as a subscriber ← confirms the rebuildworked
- Head angle changes between 78 (Alpha), 90 (Beta), 100 (Theta) ← confirmsmode routing works




---
root@elida:/workspace# ros2 node info /cara_behavior_node
/cara_behavior_node
  Subscribers:
    /cara/emotion_state: std_msgs/msg/String
    /cara/mind_mode: std_msgs/msg/String
  Publishers:
    /cara/behavior_cmd: std_msgs/msg/String
    /parameter_events: rcl_interfaces/msg/ParameterEvent
    /rosout: rcl_interfaces/msg/Log
  Service Servers:
    /cara_behavior_node/describe_parameters: rcl_interfaces/srv/DescribeParameters
    /cara_behavior_node/get_parameter_types: rcl_interfaces/srv/GetParameterTypes
    /cara_behavior_node/get_parameters: rcl_interfaces/srv/GetParameters
    /cara_behavior_node/list_parameters: rcl_interfaces/srv/ListParameters
    /cara_behavior_node/set_parameters: rcl_interfaces/srv/SetParameters
    /cara_behavior_node/set_parameters_atomically: rcl_interfaces/srv/SetParametersAtomically
  Service Clients:

  Action Servers:
---
 — behavior_node has a timer that fires every 0.4 seconds and continuously publishes to /cara/behavior_cmd. It's designed to keep streaming head angle + blink commands to the hardware at a steady rate.

ros2 topic echo just streams everything it sees on the topic until you stop it.

Hit Ctrl+C to stop the echo. If you only want to see one message, use:

ros2 topic echo /cara/behavior_cmd --once

The continuous publishing is actually correct behavior — the servo driver and Arduino bridge expect a steady stream of commands, not one-shot messages. If it stopped publishing, the hardware would have no position to hold.
---


 ### The full end-to-end test with 3 terminals, all inside the container:

---
Terminal 1 — ROS vision stack:
source /workspace/ros2_ws/install/setup.bash
ros2 launch cara_vision_control cara_runtime.launch.py
This brings up: camera → face_yunet_node → emotion_node → behavior_node → arduino_bridge_node

---
Terminal 2 — Cara brain:
source /workspace/ros2_ws/install/setup.bash
cd /workspace/app
CARA_MIC=plughw:0,0 python3 cara.py
This starts the ROS listener, loads Gemini, and opens the mic.

---
Terminal 3 — Monitor (run these one at a time):
source /workspace/ros2_ws/install/setup.bash

# Is the emotion node seeing a face?
ros2 topic echo /cara/emotion

# Is cara_main picking it up and publishing a mode?
ros2 topic echo /cara/mind_mode

# Is behavior_node responding with head commands?
ros2 topic echo /cara/behavior_cmd

# All active topics at once (sanity check)
ros2 topic list

---
What a healthy pipeline looks like:

/cara/emotion      →  "neutral (0.72)"   (streaming from emotion_node)
/cara/mind_mode    →  "BETA_REASONING"   (published by cara_main after each turn)
/cara/behavior_cmd →  "HEAD:90,BLINK:0"  (streaming from behavior_node at 2.5Hz)

/cara/mind_mode will only appear after you speak to Cara for the first time (it's published per-turn, not continuously). Everything else should stream immediately once a face is in frame.
