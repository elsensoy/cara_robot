// cara_control — the non-RL prototype of Cara's Jetson-side signal path:
//
//   BNO055 ─► ImuGuard  ─┐
//                        ├─► ObservationBuilder ─► Controller ─► SafetyFilter ─► servos
//   INA219 ─► HealthEstimator ┘        (hand-written now, learned policy later)
//   (+ per-joint INA219s, once wired, feed HealthEstimator's per-joint path)
//
// ImuGuard and HealthEstimator's telemetry_state are the "can I trust this
// sensor" layer: staleness, frozen/derivative-implausible data, and a
// commanded-vs-measured motion cross-check, all behind a persistence gate so
// one dropped read doesn't flip a verdict either way. See diagnostics.hpp.
//
// Run `--sim` (default) to watch the controller respond to a scripted actuator-
// health fault with no hardware attached. Run `--hw --serial /dev/ttyUSB0` on Cara.
// `--test-imu-freeze` / `--test-power-dropout` inject sim-only faults that
// exercise the new diagnostics end to end without touching real hardware.

#include "cara_control/diagnostics.hpp"
#include "cara_control/pipeline.hpp"
#include "cara_control/sources.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace cara;

namespace {

// Parses "0x41,0x42,66" -> {0x41, 0x42, 66}. Empty string -> empty vector.
std::vector<int> parseAddrList(const std::string& csv) {
    std::vector<int> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        out.push_back(static_cast<int>(std::strtol(tok.c_str(), nullptr, 0)));
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    bool        hw          = false;
    std::string serial_port;
    int         i2c_bus     = 1;
    int         ina_addr    = 0x45;   // servo rail, per tests/cara_power_monitor.py
    int         imu_addr    = 0x28;
    int         baud        = 115200;
    double      rate_hz     = 50.0;   // README control frequency
    double      duration_s  = 0.0;    // 0 = run forever
    std::string per_joint_addrs_csv;  // empty = per-joint current sensing not wired
    bool        test_imu_freeze    = false;
    bool        test_power_dropout = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* dflt) -> const char* {
            return (i + 1 < argc) ? argv[++i] : dflt;
        };
        if      (a == "--hw")               hw = true;
        else if (a == "--sim")              hw = false;
        else if (a == "--serial")           serial_port = next("");
        else if (a == "--rate")             rate_hz    = std::atof(next("50"));
        else if (a == "--duration")         duration_s = std::atof(next("0"));
        else if (a == "--bus")              i2c_bus    = std::atoi(next("1"));
        else if (a == "--ina-addr")         ina_addr   = static_cast<int>(std::strtol(next("0x45"), nullptr, 0));
        else if (a == "--imu-addr")         imu_addr   = static_cast<int>(std::strtol(next("0x28"), nullptr, 0));
        else if (a == "--per-joint-addrs")  per_joint_addrs_csv = next("");
        else if (a == "--test-imu-freeze")  test_imu_freeze = true;
        else if (a == "--test-power-dropout") test_power_dropout = true;
        else if (a == "--help") {
            std::puts("cara_control [--sim|--hw] [--serial <port>] [--rate HZ] [--duration S]\n"
                      "             [--bus N] [--ina-addr 0xNN] [--imu-addr 0xNN]\n"
                      "             [--per-joint-addrs 0xNN,0xNN,...]   (7 addrs, one per servo; hw only)\n"
                      "             [--test-imu-freeze] [--test-power-dropout]   (sim only)");
            return 0;
        }
    }

    const std::vector<int> per_joint_addrs = parseAddrList(per_joint_addrs_csv);
    if (hw && !per_joint_addrs.empty() && static_cast<int>(per_joint_addrs.size()) != NUM_SERVOS) {
        std::fprintf(stderr, "--per-joint-addrs needs exactly %d addresses, got %zu\n",
                     NUM_SERVOS, per_joint_addrs.size());
        return 1;
    }

    std::unique_ptr<ImuSource>           imu_src;
    std::unique_ptr<PowerSource>         pwr_src;
    std::unique_ptr<PerJointPowerSource> pj_src;
    std::unique_ptr<SetpointSource>      set_src = makeGaitSetpoint();
    std::unique_ptr<ServoOutput>         out_dev;

    try {
        if (hw) {
            imu_src = makeBno055Imu(i2c_bus, imu_addr);
            pwr_src = makeIna219Power(i2c_bus, ina_addr);
            pj_src  = per_joint_addrs.empty() ? makeNullPerJointPower()
                                              : makeIna219MultiPower(i2c_bus, per_joint_addrs);
        } else {
            imu_src = makeSimImu();
            pwr_src = makeSimPower();
            pj_src  = makeSimPerJointPower();   // free demo data, no flag needed
            if (test_imu_freeze)    setSimImuFreezeTest(true);
            if (test_power_dropout) setSimPowerDropoutTest(true);
        }
        out_dev = serial_port.empty() ? makeConsoleOutput()
                                      : makeSerialOutput(serial_port, baud);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "startup failed: %s\n", e.what());
        return 1;
    }
    if (hw && (test_imu_freeze || test_power_dropout)) {
        std::fprintf(stderr, "note: --test-imu-freeze/--test-power-dropout only affect --sim; ignored under --hw\n");
    }

    HealthEstimator        health;
    ObservationBuilder     obs_builder;
    HandwrittenController  controller;
    SafetyFilter           safety;
    ImuGuard                imu_guard;

    Observation obs;
    Action      action;   // also doubles as "last commanded action" for ImuGuard's motion cross-check

    const double dt_nom = 1.0 / rate_hz;
    double t_prev = now_s();
    double t_next = t_prev;
    double t_log  = -1.0;

    std::printf("cara_control: %s mode | %.0f Hz | output=%s | per-joint current=%s\n",
                hw ? "HARDWARE" : "SIM", rate_hz,
                serial_port.empty() ? "console" : serial_port.c_str(),
                (hw && per_joint_addrs.empty()) ? "not wired" : "on");
    std::printf("%7s  %6s  %-8s  %-8s  %-4s  %8s  %6s  %6s  %6s  %-4s   action[0..6] (rad)\n",
                "t", "health", "sev", "tlm", "imu", "I(mA)", "V(V)", "grav_z", "gain", "pj");

    while (true) {
        const double t = now_s();
        const float dt = static_cast<float>(t - t_prev);
        t_prev = t;

        const PowerSample           ps      = pwr_src->read();
        const PerJointCurrentSample pjs     = pj_src->read();
        const ImuSample             is_raw  = imu_src->read();
        // `action` still holds the previous tick's committed, safety-filtered
        // command here -- exactly what's actually driving the servos right
        // now, which is what this reading's motion should be checked against.
        const ImuSample              is     = imu_guard.update(is_raw, action, t);
        const ServoCommandSample     sp     = set_src->read(t);

        const HealthState& hs = health.update(ps, &pjs);
        obs_builder.build(is, sp, hs, obs);
        controller.compute(obs, sp, action);
        safety.apply(dt > 0.f ? dt : static_cast<float>(dt_nom), action);
        out_dev->write(action);

        if (t - t_log >= 0.2) {   // ~5 Hz human-readable log
            t_log = t;
            const auto& a = action.target_rad;
            const auto& d = imu_guard.diagnostics();
            float pj_min = 1.f;
            for (int j = 0; j < NUM_SERVOS; ++j) pj_min = std::min(pj_min, hs.per_servo[j]);
            std::printf("%7.2f  %6.2f  %-8s  %-8s  %-4s  %8.0f  %6.2f  %6.2f  %6.2f  %4.2f   "
                        "[% .3f % .3f % .3f % .3f % .3f % .3f % .3f]\n",
                        t, hs.system, hs.label, hs.telemetry_state, d.state,
                        hs.current_ema_ma, hs.voltage_ema_v,
                        obs.projected_grav()[2], controller.last_gain(),
                        hs.per_servo_valid ? pj_min : -1.f,
                        a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
            if (d.frozen_suspected) std::fprintf(stderr, "[imu] frozen data suspected (seq=%u)\n", is_raw.seq);
            if (d.motion_mismatch)  std::fprintf(stderr, "[imu] commanded-still but measured motion (seq=%u)\n", is_raw.seq);
            if (std::strcmp(hs.telemetry_state, "fault") == 0)
                std::fprintf(stderr, "[power] telemetry fault -- health decaying toward floor\n");
        }

        if (duration_s > 0.0 && t >= duration_s) break;

        t_next += dt_nom;
        const double sleep_s = t_next - now_s();
        if (sleep_s > 0.0)
            std::this_thread::sleep_for(std::chrono::duration<double>(sleep_s));
        else
            t_next = now_s();   // fell behind — resync rather than spiral
    }

    return 0;
}
