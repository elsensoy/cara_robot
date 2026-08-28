
# ARCHITECTURE FOR CARA
```
Jetson Nano / ROS 2 brain
   |
   | USB serial 1
   v
Arduino upper-body controller
   |
   v
PCA9685 upper body
   |
   v
arms, neck, torso

Jetson Nano / ROS 2 brain
   |
   | USB serial 2
   v
Arduino lower-body controller
   |
   v
PCA9685 lower body
   |
   v
legs
```

 Upper body and lower body can be tested independently.

#### Power:
```
12 V battery
   |
   |--> 5.5–6 V buck, 8–10 A -> upper body servos
   |
   |--> 5.5–6 V buck, 15–20 A -> lower body servos
```
#### Common ground:

upper buck GND
lower buck GND
upper Arduino GND
lower Arduino GND
Jetson GND through USB

### LEGS
#### Start with:

4-DOF legs: left/right hip pitch + left/right knee pitch

Then iterate to:

6-DOF legs: add ankle roll or ankle pitch

#### And before RL or AI walking:

mechanical limits
power rails
static poses
weight shifting
supported stepping
fall safety
then learning/control

### On Jetson

```
PCA9685 #1: upper body
CH0 left shoulder
CH1 right shoulder
CH2 left arm
CH3 right arm
CH4 hip/skeleton
CH5 neck yaw
CH6 neck pitch

PCA9685 #2: lower body
CH0 left hip
CH1 left knee
CH2 left ankle / foot tilt
CH3 right hip
CH4 right knee
CH5 right ankle / foot tilt


Jetson ROS 2
   |
   | USB serial
   v
Arduino Nano / or future microcontroller
   |
   | I2C
   v
PCA9685 upper body + PCA9685 lower body

```
setup: 
```
Arduino USB -> computer(jetson)
Arduino 5V  -> PCA9685 VCC
Arduino GND -> PCA9685 GND
Arduino A4  -> PCA9685 SDA
Arduino A5  -> PCA9685 SCL

Buck OUT+ -> PCA9685 V+
Buck OUT- -> PCA9685 GND
Capacitor + -> V+
Capacitor - -> GND

CH0 -> left shoulder MG90S
CH1 -> right shoulder MG90S
CH2 -> left arm MG90S
CH3 -> right arm MG90S
CH4 -> MG996R hip

#define LEFT_SHOULDER   0
#define RIGHT_SHOULDER  1
#define LEFT_ARM        2
#define RIGHT_ARM       3
#define HIP             4
#define NECK_YAW        5
#define NECK_PITCH      6
```
CH0–CH4 = torso/arms/hip
CH5–CH6 = neck
CH7+    = future eyes, ears, eyelids, tail, etc.

 

Let's put multiple PCA9685 boards on the same I2C bus by giving them different addresses. For example:
```
Adafruit_PWMServoDriver upper = Adafruit_PWMServoDriver(0x40);
Adafruit_PWMServoDriver lower = Adafruit_PWMServoDriver(0x41);
```

CH0 left shoulder: neutral = 90, safe min = ?, safe max = ?
CH1 right shoulder: neutral = 90, safe min = ?, safe max = ?
CH2 left arm: neutral = 90, safe min = ?, safe max = ?
CH3 right arm: neutral = 90, safe min = ?, safe max = ?
CH4 hip/skeleton MG996R: neutral = 90, safe min = ?, safe max = ?
```
## LOCAL POWER SETUP
```
                 12V Battery
                +          -
                |          |
                v          v
          Buck Converter 12V -> 5.5/6V
                +          -
                |          |
                |          +----------------------+
                |                                 |
                v                                 v
          PCA9685 V+                         PCA9685 GND
                |                                 |
         Servo red wires                  Servo brown/black wires
                                                  |
Arduino Nano GND ---------------------------------+

Arduino Nano 5V  -> PCA9685 VCC
Arduino Nano A4  -> PCA9685 SDA
Arduino Nano A5  -> PCA9685 SCL

PCA9685 PWM channels:
CH0 -> MG90S
CH1 -> MG90S
CH2 -> MG90S
CH3 -> MG90S
CH4 -> MG996R
```
4 MG90S  × ~0.8 A  = ~3.2 A
1 MG996R × ~2.5 A  = ~2.5 A
2 SG90   × ~0.6 A  = ~1.2 A
----------------------------
Worst case total    = ~6.9 A
### Arduino Nano controls one MG90S through PCA9685 using external 5–6V servo power.

### Next test(completed)
Add 4 MG90S.
Then test MG996R alone.
Then combine all 5.
Then make simple limb poses.
Then connect Arduino to Jetson over serial.
Then replace Arduino control with ROS 2 commands.

For the eventual ROS 2 setup, probably make the Arduino a simple servo command interpreter first. Jetson sends something like:

```
S0:90
S1:110
S2:70
S3:90
S4:95
```

### Set it up in two separate systems:

Logic/control system: Arduino Nano -> PCA9685
Servo power system: Battery -> buck converter -> PCA9685 servo power rail

Do not power the servos from the Arduino

1. Parts so  need

so  already have:

Arduino Nano
PCA9685 servo driver
4 × MG90S servos
1 × MG996R servo
12V Yalentcell battery
1000 µF 25V capacitor
5V ↔ 3.3V level shifter

so  still need, or need to confirm so  have:

DC buck converter: 12V input -> 5V or 6V output
Current rating: at least 6A, preferably 8A–10A

The buck converter is very important. so r servos should not receive 12V.

2. Power wiring

so r battery goes into the buck converter:

12V battery +  -> buck converter IN+
12V battery -  -> buck converter IN-

Set the buck converter output to around:

5.5V or 6.0V

Then connect buck output to the PCA9685 servo power terminals:

buck OUT+ -> PCA9685 V+
buck OUT- -> PCA9685 GND

The PCA9685 board usually has a terminal block labeled something like:

V+   GND

That is where servo power goes.

3. Add the capacitor

Put the capacitor across the PCA9685 servo power rail:

capacitor + -> PCA9685 V+
capacitor - -> PCA9685 GND

Be very careful with polarity. The stripe on the capacitor usually marks the negative side.

Our capacitor is rated 25V, so it is fine for a 5–6V servo rail.

4. Arduino to PCA9685 wiring

For initial Arduino Nano control, wire it like this:

Arduino 5V  -> PCA9685 VCC
Arduino GND -> PCA9685 GND
Arduino A4  -> PCA9685 SDA
Arduino A5  -> PCA9685 SCL

Then make sure the servo power ground and Arduino ground are connected together:

buck OUT- / PCA9685 GND / Arduino GND all connected

This common ground is required. Otherwise the PWM signal does not have a shared reference.

5. Servo connections

Plug the servos into the PCA9685 channels.

Example:

Channel 0 -> left shoulder MG90S
Channel 1 -> right shoulder MG90S
Channel 2 -> left arm MG90S
Channel 3 -> right arm MG90S
Channel 4 -> hip/skeleton MG996R

Servo wires are usually:

Brown/Black -> GND
Red         -> V+
Orange/Yellow/White -> signal

On the PCA9685, each channel usually has three pins:

GND | V+ | PWM

Make sure the brown/black wire goes to GND, red goes to V+, and orange/yellow goes to PWM/signal.

6. First safe test

Do not connect all servos at first.

Start with one MG90S on channel 0.

Install the Arduino library:

Adafruit PWM Servo Driver Library

Then upload this test code:

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVO_FREQ 50

// Conservative pulse range for most hobby servos.
// so  may need to tune these.
#define SERVOMIN 120
#define SERVOMAX 520

int angleToPulse(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

void setServoAngle(uint8_t channel, int angle) {
  angle = constrain(angle, 0, 180);
  int pulse = angleToPulse(angle);
  pwm.setPWM(channel, 0, pulse);
}

void setup() {
  Serial.begin(9600);
  Serial.println("PCA9685 servo test");

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);

  delay(500);
}

void loop() {
  setServoAngle(0, 60);
  delay(1000);

  setServoAngle(0, 90);
  delay(1000);

  setServoAngle(0, 120);
  delay(1000);

  setServoAngle(0, 90);
  delay(1000);
}

This avoids extreme 0° and 180° positions at first, which is safer mechanically.

7. Add servos one by one

After channel 0 works, add the other MG90S servos:

void loop() {
  setServoAngle(0, 90);
  setServoAngle(1, 90);
  setServoAngle(2, 90);
  setServoAngle(3, 90);
  delay(1000);

  setServoAngle(0, 70);
  setServoAngle(1, 110);
  setServoAngle(2, 70);
  setServoAngle(3, 110);
  delay(1000);
}

Test the MG996R separately before combining it with the arms. It can draw much more current and move with much more force.

8. Add the MG996R safely

For the hip/skeleton servo, start with a limited range:

void loop() {
  setServoAngle(4, 80);
  delay(1000);

  setServoAngle(4, 90);
  delay(1000);

  setServoAngle(4, 100);
  delay(1000);

  setServoAngle(4, 90);
  delay(1000);
}

Do not immediately swing it from 0° to 180°. That can cause a current spike or damage the skeleton mechanism.

9. Where the level shifter fits

For Arduino Nano first, so  probably do not need the 5V -> 3.3V level shifter.

Use:

Arduino Nano 5V logic -> PCA9685 VCC at 5V

Later, when using the Jetson Nano, so  should use 3.3V-safe I2C:

Jetson 3.3V -> PCA9685 VCC
Jetson SDA/SCL -> PCA9685 SDA/SCL
Jetson GND -> PCA9685 GND

or use so r level shifter:

Jetson 3.3V side -> LV side of level shifter
PCA9685/Arduino 5V side -> HV side of level shifter

But for the first Arduino test, keep it simple.

