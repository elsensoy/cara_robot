#include "cara_control/diagnostics.hpp"

#include <algorithm>
#include <cmath>

namespace cara {

namespace {

// Bit-for-bit comparison of the fields that actually vary sample to sample.
// Deliberately not a memcmp of the whole struct -- t_s/seq/valid change every
// call regardless, and comparing them would defeat the point.
bool sameOrientation(const ImuSample& a, const ImuSample& b) {
    return a.roll_rad  == b.roll_rad  &&
           a.pitch_rad == b.pitch_rad &&
           a.yaw_rad   == b.yaw_rad   &&
           a.ang_vel_rad_s[0] == b.ang_vel_rad_s[0] &&
           a.ang_vel_rad_s[1] == b.ang_vel_rad_s[1] &&
           a.ang_vel_rad_s[2] == b.ang_vel_rad_s[2];
}

} // namespace

ImuSample ImuGuard::update(const ImuSample& raw, const Action& cmd, double now) {
    diag_ = ImuDiagnostics{};

    // -- staleness: the read succeeded but the timestamp it carries is old --
    const bool stale = raw.valid && (now - raw.t_s) > cfg_.max_age_s;
    diag_.stale = stale;

    // -- frozen / derivative-implausibility: real sensor noise doesn't repeat --
    bool frozen = false;
    if (raw.valid && have_last_raw_ && last_raw_.valid && sameOrientation(raw, last_raw_)) {
        if (++frozen_repeat_ >= cfg_.frozen_repeat_trip) frozen = true;
    } else {
        frozen_repeat_ = 0;
    }
    diag_.frozen_suspected = frozen;
    if (raw.valid) { last_raw_ = raw; have_last_raw_ = true; }

    // -- commanded-vs-measured motion cross-check --
    bool mismatch = false;
    if (have_last_cmd_) {
        float max_step = 0.f;
        for (int j = 0; j < NUM_SERVOS; ++j)
            max_step = std::max(max_step, std::fabs(cmd.target_rad[j] - last_cmd_.target_rad[j]));
        const bool commanded_still = max_step < cfg_.still_cmd_eps_rad;

        if (commanded_still) {
            if (still_since_s_ < 0.0) still_since_s_ = now;
        } else {
            still_since_s_ = -1.0;
        }

        if (commanded_still && still_since_s_ >= 0.0 &&
            (now - still_since_s_) >= cfg_.still_hold_s && raw.valid) {
            const auto& w = raw.ang_vel_rad_s;
            const float wobble = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
            mismatch = wobble > cfg_.resting_wobble_rad_s;
        }
    }
    last_cmd_ = cmd;
    have_last_cmd_ = true;
    diag_.motion_mismatch = mismatch;

    // -- fold everything into one persistence-gated verdict --
    const bool bad = !raw.valid || stale || frozen || mismatch;
    const auto gstate = gate_.update(bad);
    diag_.state = PersistenceGate::label(gstate);

    if (gstate == PersistenceGate::State::Fault) {
        ImuSample out;         // default-constructed: valid=false, safe zeros
        out.t_s = raw.t_s;
        out.seq = raw.seq;
        return out;            // ObservationBuilder's existing !valid fallback takes over
    }
    if (raw.valid) { last_good_ = raw; have_last_good_ = true; return raw; }
    if (have_last_good_) return last_good_;   // degraded, within grace: hold last good
    return raw;                                // nothing good seen yet
}

} // namespace cara
