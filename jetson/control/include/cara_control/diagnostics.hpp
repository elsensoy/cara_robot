#pragma once
// Fault detection sitting between raw sensor sources and the rest of the
// pipeline. Three independent checks, folded through one hysteresis gate so
// a single bad tick can't flip a trust verdict either way:
//   - staleness        (sample older than a max age)
//   - frozen data       (derivative implausibility -- real sensor noise
//                        doesn't repeat bit-for-bit)
//   - motion mismatch   (commanded "hold still", IMU keeps reporting motion
//                        -- the servo command is a second, independent
//                        prediction of how the robot should be moving)

#include "cara_control/types.hpp"

namespace cara {

// Generic consecutive-bad / consecutive-good hysteresis. Trips FAULT after
// `trip` consecutive bad ticks; clears back to OK only after `clear`
// consecutive good ticks. Deliberately asymmetric: quick to distrust, slow
// to trust again -- replaces reacting to a single dropped read with a real
// persistence threshold.
class PersistenceGate {
public:
    enum class State { Ok, Degraded, Fault };
    struct Config { int trip = 3; int clear = 10; };

    PersistenceGate() = default;
    explicit PersistenceGate(Config cfg) : cfg_(cfg) {}

    State update(bool bad) {
        if (bad) {
            if (++bad_count_ >= cfg_.trip) state_ = State::Fault;
            else if (state_ != State::Fault) state_ = State::Degraded;
            good_count_ = 0;
        } else {
            bad_count_ = 0;
            if (state_ == State::Fault) {
                if (++good_count_ >= cfg_.clear) state_ = State::Ok;
            } else {
                state_ = State::Ok;
            }
        }
        return state_;
    }

    State state() const { return state_; }
    bool  faulted() const { return state_ == State::Fault; }

    static const char* label(State s) {
        switch (s) {
            case State::Ok:       return "ok";
            case State::Degraded: return "degraded";
            default:               return "fault";
        }
    }

private:
    Config cfg_{};
    State  state_      = State::Ok;
    int    bad_count_  = 0;
    int    good_count_ = 0;
};

// Sits between the raw ImuSource and ObservationBuilder. Returns the sample
// the rest of the pipeline should act on: the fresh raw sample when healthy,
// the last known-good sample through a short grace period (so one transient
// I2C blip doesn't immediately zero out the stability signal), or an
// explicitly-invalid sample once the gate trips FAULT (ObservationBuilder's
// existing `!imu.valid` fallback then takes over, unchanged).
class ImuGuard {
public:
    struct Config {
        double max_age_s            = 0.25;  // sample older than this is stale
        int    frozen_repeat_trip   = 5;      // consecutive bit-identical samples -> "frozen"
        float  still_cmd_eps_rad    = 0.01f;  // commanded per-joint step below this = "holding still"
        float  resting_wobble_rad_s = 0.35f;  // gyro magnitude above this while "still" is suspicious
        double still_hold_s         = 0.3;    // how long "commanded still" must persist before arming
        PersistenceGate::Config gate{3, 10};
    };

    ImuGuard() : gate_(cfg_.gate) {}
    explicit ImuGuard(Config cfg) : cfg_(cfg), gate_(cfg.gate) {}

    ImuSample update(const ImuSample& raw, const Action& cmd, double now);

    const ImuDiagnostics& diagnostics() const { return diag_; }

private:
    Config          cfg_;
    PersistenceGate gate_;
    ImuDiagnostics  diag_{};

    ImuSample last_good_{};
    bool      have_last_good_ = false;
    ImuSample last_raw_{};
    bool      have_last_raw_  = false;
    int       frozen_repeat_  = 0;

    Action last_cmd_{};
    bool   have_last_cmd_ = false;
    double still_since_s_ = -1.0;
};

} // namespace cara
