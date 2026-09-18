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

// Duration-based bad/good hysteresis. Trips FAULT once bad readings have
// persisted continuously for `trip_s`; clears back to OK only once good
// readings have persisted continuously for `clear_s`. Deliberately
// asymmetric: quick to distrust, slow to trust again.
//
// Time-based on purpose, not tick-counted: this loop nominally runs at a
// fixed rate, but "3 consecutive bad ticks" silently redefines itself every
// time real scheduling jitter changes how long a tick actually takes -- a
// single slow iteration (see the timing cluster: what happens if one
// iteration takes 80ms) would make 3 ticks cover far more wall-clock time
// than intended, and a burst of fast ticks would make it cover far less.
// Driving the gate from the caller's own monotonic timestamp instead keeps
// fault semantics defined in real time, independent of how fast the loop
// happens to be running at that moment.
class PersistenceGate {
public:
    enum class State { Ok, Degraded, Fault };
    struct Config { double trip_s = 0.06; double clear_s = 0.20; };

    PersistenceGate() = default;
    explicit PersistenceGate(Config cfg) : cfg_(cfg) {}

    // `now` must be monotonic (the same clock as the samples' own
    // timestamps) -- not wall-clock-of-day, which can jump.
    State update(bool bad, double now) {
        if (bad) {
            if (bad_since_ < 0.0) bad_since_ = now;
            good_since_ = -1.0;
            if (now - bad_since_ >= cfg_.trip_s) state_ = State::Fault;
            else if (state_ != State::Fault) state_ = State::Degraded;
        } else {
            bad_since_ = -1.0;
            if (state_ == State::Fault) {
                if (good_since_ < 0.0) good_since_ = now;
                if (now - good_since_ >= cfg_.clear_s) state_ = State::Ok;
            } else {
                good_since_ = -1.0;
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
    double bad_since_  = -1.0;   // monotonic time the current bad streak started, or -1
    double good_since_ = -1.0;   // monotonic time the current (post-fault) good streak started, or -1
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
        PersistenceGate::Config gate{0.06, 0.20};   // seconds: trip / clear (see PersistenceGate)
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
