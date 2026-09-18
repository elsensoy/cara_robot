#include "cara_control/sources.hpp"

#include <chrono>
#include <cmath>

namespace cara {

double now_s() {
    using namespace std::chrono;
    static const auto t0 = steady_clock::now();
    return duration<double>(steady_clock::now() - t0).count();
}

namespace {

// Set >= 0 by setSimFaultOverride() to force a fault level; < 0 uses the timeline.
float g_fault_override = -1.f;

// Diagnostics test-injection toggles -- off by default, see sources.hpp.
bool g_imu_freeze_test    = false;
bool g_power_dropout_test = false;

// One shared fault timeline so the simulated IMU and power source tell a
// consistent story: a ~4 s window of servo-rail overdraw + voltage sag, once
// per 16 s. Returns 0 (nominal) .. 1 (full fault).
float faultLevel(double t) {
    if (g_fault_override >= 0.f)
        return g_fault_override > 1.f ? 1.f : g_fault_override;
    const double p = std::fmod(t, 16.0);
    if (p <  4.0) return 0.f;
    if (p <  8.0) return static_cast<float>((p - 4.0) / 4.0);    // ramp in
    if (p < 11.0) return 1.f;                                    // hold
    if (p < 15.0) return static_cast<float>((15.0 - p) / 4.0);   // ramp out
    return 0.f;
}

struct SimPower : PowerSource {
    std::uint32_t seq_ = 0;

    PowerSample read() override {
        const double t = now_s();
        PowerSample s;
        s.t_s = t;
        s.seq = seq_++;

        // Dropout-test window: report a dead sensor for part of each 16s
        // cycle, same cadence as the fault timeline, so a demo run shows
        // HealthEstimator's telemetry_state escalate to "fault" (after the
        // persistence gate's trip count) and recover on its own.
        if (g_power_dropout_test && std::fmod(t, 16.0) < 6.0) {
            s.valid = false;
            return s;
        }

        const float f = faultLevel(t);
        s.valid = true;
        s.current_ma    = 240.f + f * 2500.f
                        + 20.f * std::sin(t * 7.0)
                        + (f > 0.f ? 60.f * std::sin(t * 31.0) : 0.f);
        s.bus_voltage_v = 5.05f - f * 0.55f - 0.01f * std::sin(t * 5.0);
        return s;
    }
};

struct SimImu : ImuSource {
    std::uint32_t seq_           = 0;
    bool          frozen_latched_ = false;
    ImuSample     frozen_value_{};

    static ImuSample computeNominal(double t) {
        const float amp = 0.04f + 0.06f * faultLevel(t);   // strained gait sways more
        ImuSample s;
        s.t_s   = t;
        s.valid = true;
        s.roll_rad  = amp * std::sin(t * 2.0);
        s.pitch_rad = amp * std::sin(t * 1.3 + 0.5);
        s.yaw_rad   = 0.f;
        s.ang_vel_rad_s[0] = amp * 2.0f * std::cos(t * 2.0);
        s.ang_vel_rad_s[1] = amp * 1.3f * std::cos(t * 1.3 + 0.5);
        s.ang_vel_rad_s[2] = 0.f;
        return s;
    }

    ImuSample read() override {
        const double t = now_s();
        ImuSample s;

        // Freeze-test window: latch the value from the instant the window
        // opened and keep returning that exact sample -- bit-identical, on
        // purpose -- so a demo run shows ImuGuard's frozen-data detector
        // trip (after frozen_repeat_trip consecutive ticks) and recover once
        // the window closes.
        if (g_imu_freeze_test && std::fmod(t, 16.0) < 6.0) {
            if (!frozen_latched_) { frozen_value_ = computeNominal(t); frozen_latched_ = true; }
            s = frozen_value_;
        } else {
            frozen_latched_ = false;
            s = computeNominal(t);
        }
        s.t_s = t;
        s.seq = seq_++;
        return s;
    }
};

struct GaitSetpoint : SetpointSource {
    ServoCommandSample read(double t_s) override {
        ServoCommandSample c;
        c.t_s = t_s;
        const float ph = 2.f * kPi * 0.5f * static_cast<float>(t_s);   // 0.5 Hz
        c.target_rad[0] =  0.20f * std::sin(ph);          // left_shoulder
        c.target_rad[1] = -0.20f * std::sin(ph);          // right_shoulder (anti-phase)
        c.target_rad[2] =  0.15f * std::sin(ph + 0.3f);   // left_arm
        c.target_rad[3] = -0.15f * std::sin(ph + 0.3f);   // right_arm
        c.target_rad[4] =  0.08f * std::sin(ph * 2.f);    // hip
        c.target_rad[5] =  0.10f * std::sin(ph * 0.5f);   // neck_yaw
        c.target_rad[6] =  0.05f * std::sin(ph);          // neck_pitch
        return c;
    }
};

struct ConsoleOutput : ServoOutput {
    void write(const Action&) override { /* human-readable logging is in main() */ }
};

// Demo per-joint currents so `--sim` can exercise the per-joint health path
// with no hardware -- split off the same aggregate fault timeline, unevenly,
// so the channels aren't just the aggregate divided by NUM_SERVOS.
struct SimPerJointPower : PerJointPowerSource {
    std::uint32_t seq_ = 0;

    PerJointCurrentSample read() override {
        const double t = now_s();
        const float  f = faultLevel(t);
        PerJointCurrentSample s;
        s.t_s = t;
        s.seq = seq_++;
        s.present = true;
        for (int j = 0; j < NUM_SERVOS; ++j) {
            const float share = 0.10f + 0.03f * static_cast<float>(j % 3);
            s.current_ma[j]    = share * (240.f + f * 2500.f) + 8.f * std::sin(t * 11.0 + j);
            s.channel_valid[j] = true;
        }
        return s;
    }
};

struct NullPerJointPower : PerJointPowerSource {
    PerJointCurrentSample read() override {
        PerJointCurrentSample s;
        s.t_s = now_s();
        s.present = false;
        return s;
    }
};

} // namespace

void setSimFaultOverride(float level)         { g_fault_override      = level; }
void setSimImuFreezeTest(bool on)             { g_imu_freeze_test     = on; }
void setSimPowerDropoutTest(bool on)          { g_power_dropout_test  = on; }

std::unique_ptr<PowerSource>         makeSimPower()         { return std::make_unique<SimPower>(); }
std::unique_ptr<ImuSource>           makeSimImu()            { return std::make_unique<SimImu>(); }
std::unique_ptr<PerJointPowerSource> makeSimPerJointPower()  { return std::make_unique<SimPerJointPower>(); }
std::unique_ptr<PerJointPowerSource> makeNullPerJointPower() { return std::make_unique<NullPerJointPower>(); }
std::unique_ptr<SetpointSource>      makeGaitSetpoint()      { return std::make_unique<GaitSetpoint>(); }
std::unique_ptr<ServoOutput>         makeConsoleOutput()     { return std::make_unique<ConsoleOutput>(); }

} // namespace cara
