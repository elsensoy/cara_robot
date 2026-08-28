#pragma once
// RAII wrapper over Linux i2c-dev + SMBus. Lifted from tests/imu_test.cpp and
// extended with the 16-bit big-endian helpers the INA219 needs.
// Only compiled when CARA_WITH_HARDWARE=1.

#if CARA_WITH_HARDWARE

#include <cstdint>
#include <stdexcept>
#include <string>

extern "C" {
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <i2c/smbus.h>
}

namespace cara {

class I2CDevice {
public:
    I2CDevice(int bus, int addr) {
        std::string path = "/dev/i2c-" + std::to_string(bus);
        fd_ = ::open(path.c_str(), O_RDWR);
        if (fd_ < 0) throw std::runtime_error("open " + path + " failed");
        if (::ioctl(fd_, I2C_SLAVE, addr) < 0) {
            ::close(fd_);
            throw std::runtime_error("I2C_SLAVE set failed");
        }
    }
    ~I2CDevice() { if (fd_ >= 0) ::close(fd_); }
    I2CDevice(const I2CDevice&)            = delete;
    I2CDevice& operator=(const I2CDevice&) = delete;

    void write8(std::uint8_t reg, std::uint8_t v) {
        if (i2c_smbus_write_byte_data(fd_, reg, v) < 0)
            throw std::runtime_error("i2c write8 failed");
    }

    std::uint8_t read8(std::uint8_t reg) {
        std::int32_t r = i2c_smbus_read_byte_data(fd_, reg);
        if (r < 0) throw std::runtime_error("i2c read8 failed");
        return static_cast<std::uint8_t>(r);
    }

    // Big-endian (MSB first on the wire) 16-bit register access — INA219 order.
    std::uint16_t read16_be(std::uint8_t reg) {
        std::uint8_t b[2];
        if (i2c_smbus_read_i2c_block_data(fd_, reg, 2, b) < 0)
            throw std::runtime_error("i2c read16 failed");
        return static_cast<std::uint16_t>((b[0] << 8) | b[1]);
    }

    void write16_be(std::uint8_t reg, std::uint16_t v) {
        // SMBus write_word sends the low byte first; pre-swap so the device
        // sees MSB first.
        std::uint16_t swapped = static_cast<std::uint16_t>((v >> 8) | (v << 8));
        if (i2c_smbus_write_word_data(fd_, reg, swapped) < 0)
            throw std::runtime_error("i2c write16 failed");
    }

    void readBlock(std::uint8_t reg, std::uint8_t len, std::uint8_t* out) {
        if (i2c_smbus_read_i2c_block_data(fd_, reg, len, out) < 0)
            throw std::runtime_error("i2c block read failed");
    }

private:
    int fd_ = -1;
};

} // namespace cara

#endif // CARA_WITH_HARDWARE
