// cara_control_node — runs the jetson/control signal path inside ROS 2 and
// publishes the actuator-health signal so you can watch it move:
//
//   ros2 topic echo /cara/health/system
//   ros2 run rqt_plot rqt_plot /cara/health/system/data /cara/control/gain/data
//
// Observation-only: it does NOT command servos, so it is safe to run alongside
// cara_stack.launch.py.
//
// Setpoint (the joint targets the controller tracks toward): by default it
// follows /joint_commands — the same trajectory_msgs/JointTrajectory the
// actuator node consumes — so the health controller sees Cara's real commanded
// motion. Set `setpoint_topic` to "" for the internal demo gait instead.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/string.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

#include "cara_control/pipeline.hpp"
#include "cara_control/sources.hpp"

namespace {
// Channel index for a joint name in the shared servo table, or -1.
int channel_for(const std::string & name) {
  for (int i = 0; i < cara::NUM_SERVOS; ++i)
    if (name == cara::kJoints[i].name) return i;
  return -1;
}
}  // namespace

class CaraControlNode : public rclcpp::Node {
public:
  CaraControlNode() : rclcpp::Node("cara_control_node") {
    const std::string source = declare_parameter<std::string>("source", "sim");
    rate_hz_ = declare_parameter<double>("rate_hz", 50.0);
    const int bus      = declare_parameter<int>("i2c_bus", 1);
    const int ina_addr = declare_parameter<int>("ina_addr", 0x45);
    const int imu_addr = declare_parameter<int>("imu_addr", 0x28);

    // Setpoint source: "" -> internal demo gait; otherwise subscribe to a
    // trajectory_msgs/JointTrajectory (default /joint_commands, the actuator
    // node's own input). Named points are absolute servo degrees and update
    // only the joints they name; unnamed points are RL-policy values in [-1, 1].
    // If the topic goes quiet for longer than setpoint_timeout_s, the controller
    // falls back to neutral.
    setpoint_topic_     = declare_parameter<std::string>("setpoint_topic", "/joint_commands");
    setpoint_timeout_s_ = declare_parameter<double>("setpoint_timeout_s", 0.5);

    if (setpoint_topic_.empty()) {
      gait_src_ = cara::makeGaitSetpoint();
      RCLCPP_INFO(get_logger(), "setpoint: internal demo gait");
    } else {
      sub_setpoint_ = create_subscription<trajectory_msgs::msg::JointTrajectory>(
        setpoint_topic_, 10,
        std::bind(&CaraControlNode::on_setpoint, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(), "setpoint: tracking '%s' (%.2fs timeout)",
                  setpoint_topic_.c_str(), setpoint_timeout_s_);
    }

    try {
      if (source == "hw") {
        imu_src_ = cara::makeBno055Imu(bus, imu_addr);
        pwr_src_ = cara::makeIna219Power(bus, ina_addr);
        hw_ = true;
      } else {
        imu_src_ = cara::makeSimImu();
        pwr_src_ = cara::makeSimPower();
      }
    } catch (const std::exception & e) {
      RCLCPP_FATAL(get_logger(), "source init failed: %s", e.what());
      throw;
    }

    pub_health_  = create_publisher<std_msgs::msg::Float32>("/cara/health/system", 10);
    pub_state_   = create_publisher<std_msgs::msg::String>("/cara/health/state", 10);
    pub_current_ = create_publisher<std_msgs::msg::Float32>("/cara/health/servo_rail_current_ma", 10);
    pub_voltage_ = create_publisher<std_msgs::msg::Float32>("/cara/health/servo_rail_voltage_v", 10);
    pub_gain_    = create_publisher<std_msgs::msg::Float32>("/cara/control/gain", 10);
    pub_obs_     = create_publisher<std_msgs::msg::Float32MultiArray>("/cara/control/observation", 10);
    pub_action_  = create_publisher<std_msgs::msg::Float32MultiArray>("/cara/control/action", 10);

    if (!hw_) {
      // Force the simulated fault level (0..1); negative releases to the timeline.
      //   ros2 topic pub --once /cara/sim/fault std_msgs/Float32 "{data: 1.0}"
      sub_fault_ = create_subscription<std_msgs::msg::Float32>(
        "/cara/sim/fault", 10,
        [this](const std_msgs::msg::Float32::SharedPtr m) {
          cara::setSimFaultOverride(m->data);
          RCLCPP_INFO(get_logger(), "sim fault override = %.2f", m->data);
        });
    }

    t_prev_ = cara::now_s();
    const std::chrono::duration<double> period(1.0 / rate_hz_);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&CaraControlNode::step, this));

    RCLCPP_INFO(get_logger(), "cara_control_node: %s mode, %.0f Hz",
                hw_ ? "HARDWARE" : "SIM", rate_hz_);
  }

private:
  // Cache the latest external setpoint. Runs in the same thread as step() under
  // the default single-threaded executor, so no lock is needed.
  void on_setpoint(const trajectory_msgs::msg::JointTrajectory::SharedPtr m) {
    if (m->points.empty()) return;
    const auto & pos = m->points.front().positions;   // points[0], as the actuator node does

    if (!m->joint_names.empty()) {
      // Named: positions are absolute servo degrees. Update only the joints
      // named; the rest hold their last value (matches cara_actuator_node).
      const std::size_t k = std::min(m->joint_names.size(), pos.size());
      for (std::size_t i = 0; i < k; ++i) {
        const int ch = channel_for(m->joint_names[i]);
        if (ch < 0) {
          RCLCPP_WARN_ONCE(get_logger(), "setpoint names unknown joint '%s'",
                           m->joint_names[i].c_str());
          continue;
        }
        sp_.target_rad[ch] = cara::servoDegToRad(static_cast<float>(pos[i]));
      }
    } else {
      // Unnamed: RL-policy values in [-1, 1]; (p+1)*90 deg about 0 == p * pi/2 rad.
      constexpr std::size_t kN = cara::NUM_SERVOS;
      if (pos.size() < kN) {
        RCLCPP_WARN_ONCE(get_logger(),
          "indexed setpoint has %zu values, need >= %zu", pos.size(), kN);
        return;
      }
      for (std::size_t i = 0; i < kN; ++i) {
        const float p = std::clamp(static_cast<float>(pos[i]), -1.f, 1.f);
        sp_.target_rad[i] = p * (cara::kPi / 2.f);
      }
    }

    sp_.t_s = cara::now_s();
    got_setpoint_ = true;
  }

  cara::ServoCommandSample current_setpoint(double t) {
    if (gait_src_) return gait_src_->read(t);
    if (got_setpoint_ && (t - sp_.t_s) <= setpoint_timeout_s_) return sp_;
    if (got_setpoint_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "setpoint stale by %.2fs — holding neutral", t - sp_.t_s);
      // Clear per-joint state so a later *partial* (named) update layers onto
      // neutral rather than resurrecting values from a previous publisher.
      sp_.target_rad.fill(0.f);
    }
    return cara::ServoCommandSample{};   // no data yet, or stale: neutral
  }

  void step() {
    const double t  = cara::now_s();
    const float  dt = static_cast<float>(t - t_prev_);
    t_prev_ = t;

    const cara::PowerSample        ps = pwr_src_->read();
    const cara::ImuSample          is = imu_src_->read();
    const cara::ServoCommandSample sp = current_setpoint(t);

    const cara::HealthState & hs = health_.update(ps);
    obs_builder_.build(is, sp, hs, obs_);
    controller_.compute(obs_, sp, action_);
    safety_.apply(dt > 0.f ? dt : static_cast<float>(1.0 / rate_hz_), action_);

    std_msgs::msg::Float32 f;
    f.data = hs.system;               pub_health_->publish(f);
    f.data = hs.current_ema_ma;       pub_current_->publish(f);
    f.data = hs.voltage_ema_v;        pub_voltage_->publish(f);
    f.data = controller_.last_gain(); pub_gain_->publish(f);

    std_msgs::msg::String s;
    s.data = hs.label;
    pub_state_->publish(s);

    std_msgs::msg::Float32MultiArray obs_msg;
    obs_msg.data.assign(obs_.data.begin(), obs_.data.end());
    pub_obs_->publish(obs_msg);

    std_msgs::msg::Float32MultiArray act_msg;
    act_msg.data.assign(action_.target_rad.begin(), action_.target_rad.end());
    pub_action_->publish(act_msg);
  }

  double rate_hz_ = 50.0;
  bool   hw_ = false;
  double t_prev_ = 0.0;

  std::string              setpoint_topic_;
  double                   setpoint_timeout_s_ = 0.5;
  cara::ServoCommandSample sp_{};              // latest external setpoint
  bool                     got_setpoint_ = false;

  std::unique_ptr<cara::ImuSource>      imu_src_;
  std::unique_ptr<cara::PowerSource>    pwr_src_;
  std::unique_ptr<cara::SetpointSource> gait_src_;   // null when a topic drives the setpoint

  cara::HealthEstimator      health_;
  cara::ObservationBuilder   obs_builder_;
  cara::HandwrittenController controller_;
  cara::SafetyFilter         safety_;
  cara::Observation          obs_;
  cara::Action               action_;

  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_health_, pub_current_, pub_voltage_, pub_gain_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_state_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_obs_, pub_action_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_fault_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr sub_setpoint_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CaraControlNode>());
  rclcpp::shutdown();
  return 0;
}
