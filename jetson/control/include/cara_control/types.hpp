#pragma once
// Shared value types for the C++ health -> observation -> control signal path.
// Units: radians / rad·s⁻¹ internally (same as the Isaac policy will use);
// degrees only at the servo-output boundary.

#include <array>
#include <cstdint>

namespace cara {

// Matches the joint table in arduino/main.cpp. This is the single compile-time
// knob for the whole pipeline; grow it toward the 20-DoF URDF later.
inline constexpr int NUM_SERVOS = 7;

struct JointSpec {
    std::uint8_t channel;
    float        min_deg;
    float        neutral_deg;
    float        max_deg;
    const char*  name;
};

inline constexpr std::array<JointSpec, NUM_SERVOS> kJoints{{
    {0, 75.f, 90.f, 105.f, "left_shoulder"},
    {1, 75.f, 90.f, 105.f, "right_shoulder"},
    {2, 75.f, 90.f, 105.f, "left_arm"},
    {3, 75.f, 90.f, 105.f, "right_arm"},
    {4, 80.f, 90.f, 100.f, "hip"},
    {5, 80.f, 90.f, 100.f, "neck_yaw"},
    {6, 85.f, 90.f,  95.f, "neck_pitch"},
}};

inline constexpr float kPi      = 3.14159265358979323846f;
inline constexpr float kDeg2Rad = kPi / 180.f;
inline constexpr float kRad2Deg = 180.f / kPi;

// Servo angle convention: 90° hardware == 0 rad joint (neutral).
inline float radToServoDeg(float rad) { return 90.f + rad * kRad2Deg; }
inline float servoDegToRad(float deg) { return (deg - 90.f) * kDeg2Rad; }

// ---- Telemetry inputs -----------------------------------------------------

struct ServoCommandSample {
    double t_s = 0.0;
    // The high-level joint setpoints we last asked for (rad). Stands in for
    // measured joint position until Cara has encoders.
    std::array<float, NUM_SERVOS> target_rad{};
};

struct ImuSample {
    double t_s   = 0.0;
    bool   valid = false;
    float  roll_rad  = 0.f;
    float  pitch_rad = 0.f;
    float  yaw_rad   = 0.f;
    std::array<float, 3> ang_vel_rad_s{};   // gyro, body frame
};

struct PowerSample {
    double t_s   = 0.0;
    bool   valid = false;
    float  bus_voltage_v = 0.f;
    float  current_ma    = 0.f;             // servo-rail aggregate
};

// ---- Health -------------------------------------------------------------

// The distinction to preserve:
//   NOW   -> `system` is the only trustworthy field: one scalar derived from the
//            aggregate servo-rail INA219.
//   LATER -> once telemetry supports per-joint attribution, fill `per_servo`
//            and set `per_servo_valid`. Consumers MUST check the flag.
struct HealthState {
    float system = 1.f;                              // 0..1
    std::array<float, NUM_SERVOS> per_servo{};       // 0..1, mirrors `system` for now
    bool  per_servo_valid = false;

    // Diagnostics (not part of the observation vector):
    float       current_ema_ma = 0.f;
    float       voltage_ema_v  = 0.f;
    const char* label          = "ok";              // ok | warn | critical
};

// ---- Observation / Action --------------------------------------------------

// Layout (matches the eventual Isaac observation ordering):
//   [ 0 .. NUM_SERVOS-1 ]  commanded joint position (rad)
//   [ +0 .. +2 ]           projected gravity, body frame (unit vector)
//   [ +3 .. +5 ]           base angular velocity (rad/s)
//   [ +6 ]                 system_health (0..1)   <-- the new channel
inline constexpr int OBS_SIZE = NUM_SERVOS + 3 + 3 + 1;

struct Observation {
    std::array<float, OBS_SIZE> data{};

    float*       joint_pos()            { return data.data(); }
    const float* joint_pos()      const { return data.data(); }
    float*       projected_grav()       { return data.data() + NUM_SERVOS; }
    const float* projected_grav() const { return data.data() + NUM_SERVOS; }
    float*       ang_vel()              { return data.data() + NUM_SERVOS + 3; }
    const float* ang_vel()        const { return data.data() + NUM_SERVOS + 3; }
    float        system_health()  const { return data[OBS_SIZE - 1]; }
};

struct Action {
    std::array<float, NUM_SERVOS> target_rad{};
};

} // namespace cara
