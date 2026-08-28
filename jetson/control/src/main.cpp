// cara_control — the non-RL prototype of Cara's Jetson-side signal path:
//
//   BNO055 ─┐
//           ├─► ObservationBuilder ─► Controller ─► SafetyFilter ─► servos
//   INA219 ─┴─► HealthEstimator ──┘        (hand-written now, learned policy later)
//
// Run `--sim` (default) to watch the controller respond to a scripted actuator-
// health fault with no hardware attached. Run `--hw --serial /dev/ttyUSB0` on Cara.

#include "cara_control/pipeline.hpp"
#include "cara_control/sources.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

using namespace cara;

int main(int argc, char** argv) {
    bool        hw          = false;
    std::string serial_port;
    int         i2c_bus     = 1;
    int         ina_addr    = 0x45;   // servo rail, per tests/cara_power_monitor.py
    int         imu_addr    = 0x28;
    int         baud        = 115200;
    double      rate_hz     = 50.0;   // README control frequency
    double      duration_s  = 0.0;    // 0 = run forever

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* dflt) -> const char* {
            return (i + 1 < argc) ? argv[++i] : dflt;
        };
        if      (a == "--hw")       hw = true;
        else if (a == "--sim")      hw = false;
        else if (a == "--serial")   serial_port = next("");
        else if (a == "--rate")     rate_hz    = std::atof(next("50"));
        else if (a == "--duration") duration_s = std::atof(next("0"));
        else if (a == "--bus")      i2c_bus    = std::atoi(next("1"));
        else if (a == "--ina-addr") ina_addr   = static_cast<int>(std::strtol(next("0x45"), nullptr, 0));
        else if (a == "--imu-addr") imu_addr   = static_cast<int>(std::strtol(next("0x28"), nullptr, 0));
        else if (a == "--help") {
            std::puts("cara_control [--sim|--hw] [--serial <port>] [--rate HZ] [--duration S]\n"
                      "             [--bus N] [--ina-addr 0xNN] [--imu-addr 0xNN]");
            return 0;
        }
    }

    std::unique_ptr<ImuSource>      imu_src;
    std::unique_ptr<PowerSource>    pwr_src;
    std::unique_ptr<SetpointSource> set_src = makeGaitSetpoint();
    std::unique_ptr<ServoOutput>    out_dev;

    try {
        if (hw) {
            imu_src = makeBno055Imu(i2c_bus, imu_addr);
            pwr_src = makeIna219Power(i2c_bus, ina_addr);
        } else {
            imu_src = makeSimImu();
            pwr_src = makeSimPower();
        }
        out_dev = serial_port.empty() ? makeConsoleOutput()
                                      : makeSerialOutput(serial_port, baud);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "startup failed: %s\n", e.what());
        return 1;
    }

    HealthEstimator      health;
    ObservationBuilder   obs_builder;
    HandwrittenController controller;
    SafetyFilter         safety;

    Observation obs;
    Action      action;

    const double dt_nom = 1.0 / rate_hz;
    double t_prev = now_s();
    double t_next = t_prev;
    double t_log  = -1.0;

    std::printf("cara_control: %s mode | %.0f Hz | output=%s\n",
                hw ? "HARDWARE" : "SIM", rate_hz,
                serial_port.empty() ? "console" : serial_port.c_str());
    std::printf("%7s  %6s  %-8s  %8s  %6s  %6s  %6s   action[0..6] (rad)\n",
                "t", "health", "state", "I(mA)", "V(V)", "grav_z", "gain");

    while (true) {
        const double t = now_s();
        const float dt = static_cast<float>(t - t_prev);
        t_prev = t;

        const PowerSample       ps = pwr_src->read();
        const ImuSample         is = imu_src->read();
        const ServoCommandSample sp = set_src->read(t);

        const HealthState& hs = health.update(ps);
        obs_builder.build(is, sp, hs, obs);
        controller.compute(obs, sp, action);
        safety.apply(dt > 0.f ? dt : static_cast<float>(dt_nom), action);
        out_dev->write(action);

        if (t - t_log >= 0.2) {   // ~5 Hz human-readable log
            t_log = t;
            const auto& a = action.target_rad;
            std::printf("%7.2f  %6.2f  %-8s  %8.0f  %6.2f  %6.2f  %6.2f   "
                        "[% .3f % .3f % .3f % .3f % .3f % .3f % .3f]\n",
                        t, hs.system, hs.label, hs.current_ema_ma, hs.voltage_ema_v,
                        obs.projected_grav()[2], controller.last_gain(),
                        a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
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
