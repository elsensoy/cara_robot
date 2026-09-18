#pragma once
// Input/output sources for the control loop. Each has a simulated and a
// hardware implementation so the exact same loop runs on a laptop and on Cara.

#include <memory>
#include <string>
#include <vector>

#include "cara_control/types.hpp"

namespace cara {

// Monotonic seconds since process start.
double now_s();

struct ImuSource {
    virtual ~ImuSource() = default;
    virtual ImuSample read() = 0;
};

struct PowerSource {
    virtual ~PowerSource() = default;
    virtual PowerSample read() = 0;
};

// Produces the high-level joint setpoints the controller tracks toward. In the
// prototype this is a scripted gait; later it can be teleop or a behaviour tree.
struct SetpointSource {
    virtual ~SetpointSource() = default;
    virtual ServoCommandSample read(double t_s) = 0;
};

struct ServoOutput {
    virtual ~ServoOutput() = default;
    virtual void write(const Action& a) = 0;
};

// Per-joint current, once wiring supports it. Always safe to call: the null
// implementation just reports `present = false` every tick.
struct PerJointPowerSource {
    virtual ~PerJointPowerSource() = default;
    virtual PerJointCurrentSample read() = 0;
};

// Override the simulated fault level (0..1). Pass a negative value to fall back
// to the internal 16 s timeline. Lets the ROS node make the fault demoable on a
// topic instead of waiting for the cycle.
void setSimFaultOverride(float level);

// Test-only injection hooks for the diagnostics added on top of the sim
// sources: with freeze-test on, SimImu latches its output for a window of
// each 16s cycle (exercises ImuGuard's frozen-data detector); with dropout-
// test on, SimPower reports `valid = false` for a window of each cycle
// (exercises HealthEstimator's telemetry persistence gate). Both off by
// default -- the ordinary sim fault timeline is unaffected either way.
void setSimImuFreezeTest(bool on);
void setSimPowerDropoutTest(bool on);

// --- simulation (always built) ---
std::unique_ptr<ImuSource>           makeSimImu();
std::unique_ptr<PowerSource>         makeSimPower();          // includes a scripted fault window
std::unique_ptr<PerJointPowerSource> makeSimPerJointPower();   // demo per-joint currents
std::unique_ptr<PerJointPowerSource> makeNullPerJointPower();  // present=false, always
std::unique_ptr<SetpointSource>      makeGaitSetpoint();
std::unique_ptr<ServoOutput>         makeConsoleOutput();     // no-op; loop does the logging

// --- hardware (throw std::runtime_error if built with CARA_WITH_HARDWARE=0) ---
std::unique_ptr<ImuSource>           makeBno055Imu(int i2c_bus, int addr);
std::unique_ptr<PowerSource>         makeIna219Power(int i2c_bus, int addr);
std::unique_ptr<PerJointPowerSource> makeIna219MultiPower(int i2c_bus, const std::vector<int>& addrs);
std::unique_ptr<ServoOutput>         makeSerialOutput(const std::string& port, int baud);

} // namespace cara
