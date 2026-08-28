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
    PowerSample read() override {
        const double t = now_s();
        const float  f = faultLevel(t);
        PowerSample s;
        s.t_s   = t;
        s.valid = true;
        s.current_ma    = 240.f + f * 2500.f
                        + 20.f * std::sin(t * 7.0)
                        + (f > 0.f ? 60.f * std::sin(t * 31.0) : 0.f);
        s.bus_voltage_v = 5.05f - f * 0.55f - 0.01f * std::sin(t * 5.0);
        return s;
    }
};

struct SimImu : ImuSource {
    ImuSample read() override {
        const double t   = now_s();
        const float  amp = 0.04f + 0.06f * faultLevel(t);   // strained gait sways more
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

} // namespace

void setSimFaultOverride(float level) { g_fault_override = level; }

std::unique_ptr<PowerSource>    makeSimPower()      { return std::make_unique<SimPower>(); }
std::unique_ptr<ImuSource>      makeSimImu()        { return std::make_unique<SimImu>(); }
std::unique_ptr<SetpointSource> makeGaitSetpoint()  { return std::make_unique<GaitSetpoint>(); }
std::unique_ptr<ServoOutput>    makeConsoleOutput() { return std::make_unique<ConsoleOutput>(); }

} // namespace cara
