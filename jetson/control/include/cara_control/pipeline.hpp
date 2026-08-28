#pragma once
// The four in-loop stages: health estimate -> observation -> controller -> safety.
// Everything here is allocation-free and runs in the control loop.

#include "cara_control/types.hpp"

namespace cara {

// Turns servo-rail INA219 telemetry into a health scalar in [0, 1].
// Port of the EMA + threshold logic in tests/cara_power_monitor.py.
class HealthEstimator {
public:
    struct Config {
        float idle_current_ma     = 250.f;    // bench baseline, all servos holding
        float warn_current_ma     = 2500.f;   // cara_power_monitor.py servo rail
        float critical_current_ma = 3100.f;
        float nominal_voltage_v   = 5.0f;
        float warn_voltage_v      = 4.8f;
        float critical_voltage_v  = 4.4f;
        float current_alpha       = 0.5f;     // matches cara_power_monitor.py
        float voltage_alpha       = 0.3f;
    };

    HealthEstimator() = default;
    explicit HealthEstimator(Config cfg) : cfg_(cfg) {}

    const HealthState& update(const PowerSample& s);
    const HealthState& state() const { return st_; }

private:
    Config      cfg_{};
    HealthState st_;
    float       i_ema_ = 0.f;
    float       v_ema_ = 0.f;
    bool        init_  = false;
};

// Assembles the policy observation vector. The health scalar is appended as the
// last element; when per-servo health becomes real, the vector extends here.
class ObservationBuilder {
public:
    void build(const ImuSample& imu, const ServoCommandSample& cmd,
               const HealthState& health, Observation& out) const;
};

class Controller {
public:
    virtual ~Controller() = default;
    virtual void compute(const Observation& obs, const ServoCommandSample& setpoint,
                         Action& out) = 0;
};

// Deliberately dumb. Passes the scripted setpoint through, but shrinks motion
// around neutral as system_health drops (and as the IMU reports base wobble).
// This is the seam an exported RL policy replaces later: same Observation in,
// same Action out.
class HandwrittenController : public Controller {
public:
    void compute(const Observation& obs, const ServoCommandSample& setpoint,
                 Action& out) override;

    float last_gain() const { return last_gain_; }   // for logging only

private:
    float last_gain_ = 1.f;
};

// Clamps to the per-joint limits from arduino/main.cpp and rate-limits each
// joint. Last line of defence before the servo command leaves the Jetson.
class SafetyFilter {
public:
    struct Config { float max_rate_rad_s = 2.5f; };

    SafetyFilter() = default;
    explicit SafetyFilter(Config cfg) : cfg_(cfg) {}

    void apply(float dt_s, Action& a);

private:
    Config                          cfg_{};
    std::array<float, NUM_SERVOS>   last_{};
    bool                            init_ = false;
};

} // namespace cara
