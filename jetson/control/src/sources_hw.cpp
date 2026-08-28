#define _DEFAULT_SOURCE
#include "cara_control/sources.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <thread>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#if CARA_WITH_HARDWARE
#include "cara_control/i2c_device.hpp"
#endif

namespace cara {

// ---- serial servo output (POSIX termios — always available) ----------------

namespace {

struct SerialOutput : ServoOutput {
    int fd_ = -1;

    SerialOutput(const std::string& port, int baud) {
        fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0) throw std::runtime_error("open " + port + " failed");

        termios tio{};
        if (tcgetattr(fd_, &tio) != 0) {
            ::close(fd_);
            throw std::runtime_error("tcgetattr failed on " + port);
        }
        cfmakeraw(&tio);
        const speed_t sp = (baud == 9600) ? B9600 : B115200;
        cfsetispeed(&tio, sp);
        cfsetospeed(&tio, sp);
        tio.c_cflag |= (CLOCAL | CREAD);
        tcsetattr(fd_, TCSANOW, &tio);

        std::this_thread::sleep_for(std::chrono::seconds(2));  // let the Nano reset
    }

    ~SerialOutput() override { if (fd_ >= 0) ::close(fd_); }

    void write(const Action& a) override {
        char buf[32];
        for (int j = 0; j < NUM_SERVOS; ++j) {
            const int deg = static_cast<int>(std::lround(radToServoDeg(a.target_rad[j])));
            const int n   = std::snprintf(buf, sizeof(buf), "S%d,%d\n", kJoints[j].channel, deg);
            if (n > 0) {
                const ssize_t w = ::write(fd_, buf, static_cast<size_t>(n));
                (void)w;
            }
        }
    }
};

} // namespace

std::unique_ptr<ServoOutput> makeSerialOutput(const std::string& port, int baud) {
    return std::make_unique<SerialOutput>(port, baud);
}

// ---- I2C sensors ----------------------------------------------------------

#if CARA_WITH_HARDWARE

namespace {

// INA219 calibrated to the Adafruit 32V / 2A config that
// tests/cara_power_monitor.py depends on (bus LSB 4 mV, current LSB 0.1 mA).
struct Ina219Power : PowerSource {
    static constexpr std::uint8_t REG_CONFIG = 0x00, REG_BUS = 0x02,
                                  REG_CURRENT = 0x04, REG_CAL = 0x05;
    I2CDevice dev_;

    Ina219Power(int bus, int addr) : dev_(bus, addr) {
        dev_.write16_be(REG_CONFIG, 0x399F);
        dev_.write16_be(REG_CAL, 4096);
    }

    PowerSample read() override {
        PowerSample s;
        s.t_s = now_s();
        try {
            dev_.write16_be(REG_CAL, 4096);   // chip quirk: refresh before current read
            const std::uint16_t braw = dev_.read16_be(REG_BUS);
            s.bus_voltage_v = static_cast<float>(braw >> 3) * 0.004f;
            const std::int16_t iraw = static_cast<std::int16_t>(dev_.read16_be(REG_CURRENT));
            s.current_ma = iraw * 0.1f;
            if (s.current_ma < 0.f) s.current_ma = 0.f;
            s.valid = true;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "INA219 read failed: %s\n", e.what());
        }
        return s;
    }
};

// BNO055 — same register map and NDOF bring-up as tests/imu_test.cpp.
struct Bno055Imu : ImuSource {
    static constexpr std::uint8_t CHIP_ID = 0x00, OPR_MODE = 0x3D, PWR_MODE = 0x3E,
                                  SYS_TRIGGER = 0x3F, EULER_H_LSB = 0x1A, GYRO_X_LSB = 0x14;
    I2CDevice dev_;

    Bno055Imu(int bus, int addr) : dev_(bus, addr) {
        if (dev_.read8(CHIP_ID) != 0xA0) throw std::runtime_error("not a BNO055");
        auto w = [&](std::uint8_t r, std::uint8_t v) {
            dev_.write8(r, v);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        };
        w(OPR_MODE, 0x00);                                       // CONFIG
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        w(PWR_MODE, 0x00);                                       // normal power
        w(SYS_TRIGGER, 0x00);
        w(OPR_MODE, 0x0C);                                       // NDOF
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    ImuSample read() override {
        ImuSample s;
        s.t_s = now_s();
        try {
            std::uint8_t e[6], g[6];
            dev_.readBlock(EULER_H_LSB, 6, e);
            dev_.readBlock(GYRO_X_LSB, 6, g);
            auto i16 = [](std::uint8_t lo, std::uint8_t hi) {
                return static_cast<std::int16_t>(lo | (hi << 8));
            };
            constexpr float D2R = kPi / 180.f;
            s.yaw_rad   = i16(e[0], e[1]) / 16.f * D2R;
            s.roll_rad  = i16(e[2], e[3]) / 16.f * D2R;
            s.pitch_rad = i16(e[4], e[5]) / 16.f * D2R;
            s.ang_vel_rad_s[0] = i16(g[0], g[1]) / 16.f * D2R;
            s.ang_vel_rad_s[1] = i16(g[2], g[3]) / 16.f * D2R;
            s.ang_vel_rad_s[2] = i16(g[4], g[5]) / 16.f * D2R;
            s.valid = true;
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "BNO055 read failed: %s\n", ex.what());
        }
        return s;
    }
};

} // namespace

std::unique_ptr<PowerSource> makeIna219Power(int i2c_bus, int addr) {
    return std::make_unique<Ina219Power>(i2c_bus, addr);
}

std::unique_ptr<ImuSource> makeBno055Imu(int i2c_bus, int addr) {
    return std::make_unique<Bno055Imu>(i2c_bus, addr);
}

#else  // !CARA_WITH_HARDWARE

std::unique_ptr<PowerSource> makeIna219Power(int, int) {
    throw std::runtime_error("built with CARA_WITH_HARDWARE=0 — reconfigure with -DCARA_WITH_HARDWARE=ON");
}

std::unique_ptr<ImuSource> makeBno055Imu(int, int) {
    throw std::runtime_error("built with CARA_WITH_HARDWARE=0 — reconfigure with -DCARA_WITH_HARDWARE=ON");
}

#endif // CARA_WITH_HARDWARE

} // namespace cara
