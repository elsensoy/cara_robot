#include "cara_control/pipeline.hpp"

#include <algorithm>
#include <cmath>

namespace cara {

static float clamp01(float x) { return std::clamp(x, 0.f, 1.f); }

// ---------------------------------------------------------------------------

const HealthState& HealthEstimator::update(const PowerSample& s, const PerJointCurrentSample* per_joint) {
    // s.t_s is set by the driver whether or not the read succeeded, so this
    // stays a meaningful dt even through a run of failed reads.
    const double dt = (t_last_call_ >= 0.0) ? std::max(0.0, s.t_s - t_last_call_) : 0.0;
    t_last_call_ = s.t_s;

    const auto gstate = telemetry_gate_.update(!s.valid);
    st_.telemetry_state = PersistenceGate::label(gstate);

    if (gstate == PersistenceGate::State::Fault) {
        // Telemetry has been down long enough that "last known good" is no
        // longer something to act on as if it were current: decay toward a
        // conservative floor instead of silently reporting the old reading
        // forever, however healthy it happened to look when it stopped.
        st_.system = std::max(cfg_.stale_health_floor,
                               static_cast<double>(st_.system) - cfg_.stale_decay_per_s * dt);
        updatePerJoint(nullptr);
        return st_;
    }
    if (!s.valid) return st_;   // degraded but not yet faulted: hold the last estimate, as before

    if (!init_) {
        i_ema_ = s.current_ma;
        v_ema_ = s.bus_voltage_v;
        init_  = true;
    } else {
        i_ema_ += cfg_.current_alpha * (s.current_ma    - i_ema_);
        v_ema_ += cfg_.voltage_alpha * (s.bus_voltage_v - v_ema_);
    }

    // 1 at idle draw, 0 at the critical overdraw threshold.
    const float ch = 1.f - (i_ema_ - cfg_.idle_current_ma) /
                           (cfg_.critical_current_ma - cfg_.idle_current_ma);
    // 1 at nominal rail voltage, 0 at the critical sag threshold.
    const float vh = (v_ema_ - cfg_.critical_voltage_v) /
                     (cfg_.nominal_voltage_v - cfg_.critical_voltage_v);

    const float h = clamp01(std::min(ch, vh));

    st_.system = h;
    st_.current_ema_ma  = i_ema_;
    st_.voltage_ema_v   = v_ema_;
    st_.label =
        (i_ema_ > cfg_.critical_current_ma || v_ema_ < cfg_.critical_voltage_v) ? "critical" :
        (i_ema_ > cfg_.warn_current_ma     || v_ema_ < cfg_.warn_voltage_v)     ? "warn"     :
                                                                                 "ok";

    updatePerJoint(per_joint, vh);
    return st_;
}

void HealthEstimator::updatePerJoint(const PerJointCurrentSample* pj, float voltage_health) {
    if (!pj || !pj->present) {
        st_.per_servo.fill(st_.system);  // mirror the scalar — NOT real attribution
        st_.per_servo_valid = false;     // flip only when a real per-joint source is wired
        return;
    }

    for (int j = 0; j < NUM_SERVOS; ++j) {
        if (pj->channel_valid[j]) {
            if (!pj_init_[j]) { pj_ema_[j] = pj->current_ma[j]; pj_init_[j] = true; }
            else pj_ema_[j] += cfg_.current_alpha * (pj->current_ma[j] - pj_ema_[j]);
        }
        // else: hold that joint's last EMA, same "dropped read holds" policy as the aggregate.
        const float ch = 1.f - (pj_ema_[j] - cfg_.idle_current_per_joint_ma) /
                               (cfg_.critical_current_per_joint_ma - cfg_.idle_current_per_joint_ma);
        st_.per_servo[j] = clamp01(std::min(ch, voltage_health));
    }
    st_.per_servo_valid = true;
}

// ---------------------------------------------------------------------------

void ObservationBuilder::build(const ImuSample& imu, const ServoCommandSample& cmd,
                               const HealthState& health, Observation& out) const {
    for (int j = 0; j < NUM_SERVOS; ++j)
        out.data[j] = cmd.target_rad[j];

    float* g = out.projected_grav();
    float* w = out.ang_vel();

    if (imu.valid) {
        const float sr = std::sin(imu.roll_rad),  cr = std::cos(imu.roll_rad);
        const float sp = std::sin(imu.pitch_rad), cp = std::cos(imu.pitch_rad);
        g[0] = -sp;          // body-frame direction of gravity (unit vector)
        g[1] =  cp * sr;
        g[2] =  cp * cr;
        w[0] = imu.ang_vel_rad_s[0];
        w[1] = imu.ang_vel_rad_s[1];
        w[2] = imu.ang_vel_rad_s[2];
    } else {
        g[0] = 0.f; g[1] = 0.f; g[2] = 1.f;
        w[0] = 0.f; w[1] = 0.f; w[2] = 0.f;
    }

    out.data[OBS_SIZE - 1] = health.system;

    // LATER: when health.per_servo_valid, append health.per_servo here and
    // grow OBS_SIZE by NUM_SERVOS.
}

// ---------------------------------------------------------------------------

void HandwrittenController::compute(const Observation& obs, const ServoCommandSample& setpoint,
                                    Action& out) {
    const float h = clamp01(obs.system_health());

    // How fast the base is rotating (rad/s) — pull motion back if Cara is wobbling.
    const float* w = obs.ang_vel();
    const float wobble = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    const float stab   = 1.f - std::clamp(wobble / 3.0f, 0.f, 0.4f);

    // Never fully freeze: keep >=25% authority so posture still holds.
    const float gain = (0.25f + 0.75f * h) * stab;
    last_gain_ = gain;

    for (int j = 0; j < NUM_SERVOS; ++j)
        out.target_rad[j] = gain * setpoint.target_rad[j];   // neutral == 0 rad
}

// ---------------------------------------------------------------------------

void SafetyFilter::apply(float dt_s, Action& a) {
    if (!init_) { last_ = a.target_rad; init_ = true; }

    const float max_step = cfg_.max_rate_rad_s * dt_s;

    for (int j = 0; j < NUM_SERVOS; ++j) {
        const float lo = servoDegToRad(kJoints[j].min_deg);
        const float hi = servoDegToRad(kJoints[j].max_deg);
        const float target = std::clamp(a.target_rad[j], lo, hi);
        const float step   = std::clamp(target - last_[j], -max_step, max_step);
        last_[j] += step;
        a.target_rad[j] = last_[j];
    }
}

} // namespace cara
