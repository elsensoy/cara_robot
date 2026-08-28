#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVO_FREQ 50
#define SERVOMIN 120
#define SERVOMAX 520

#define LEFT_SHOULDER   0
#define RIGHT_SHOULDER  1
#define LEFT_ARM        2
#define RIGHT_ARM       3
#define HIP             4
#define NECK_YAW        5
#define NECK_PITCH      6

struct JointLimit {
  uint8_t channel;
  int minAngle;
  int neutralAngle;
  int maxAngle;
  const char* name;
};

JointLimit joints[] = {
  {LEFT_SHOULDER,  75, 90, 105, "left_shoulder"},
  {RIGHT_SHOULDER, 75, 90, 105, "right_shoulder"},
  {LEFT_ARM,       75, 90, 105, "left_arm"},
  {RIGHT_ARM,      75, 90, 105, "right_arm"},
  {HIP,            80, 90, 100, "hip"},
  {NECK_YAW,       80, 90, 100, "neck_yaw"},
  {NECK_PITCH,     85, 90,  95, "neck_pitch"}
};

const int NUM_JOINTS = sizeof(joints) / sizeof(joints[0]);

int angleToPulse(int angle) {
  angle = constrain(angle, 0, 180);
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

JointLimit* getJoint(uint8_t channel) {
  for (int i = 0; i < NUM_JOINTS; i++) {
    if (joints[i].channel == channel) return &joints[i];
  }
  return nullptr;
}

void setJointAngle(uint8_t channel, int requestedAngle) {
  JointLimit* joint = getJoint(channel);
  if (joint == nullptr) return;
  int safeAngle = constrain(requestedAngle, joint->minAngle, joint->maxAngle);
  pwm.setPWM(channel, 0, angleToPulse(safeAngle));
}

void releaseAllServos() {
  for (int ch = 0; ch < 16; ch++) pwm.setPWM(ch, 0, 0);
}

void moveAllToNeutral() {
  for (int i = 0; i < NUM_JOINTS; i++) {
    setJointAngle(joints[i].channel, joints[i].neutralAngle);
    delay(100);
  }
}

// Parse and execute: S<channel>,<angle>
void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() < 4 || cmd.charAt(0) != 'S') return;

  int comma = cmd.indexOf(',');
  if (comma < 0) return;

  int channel = cmd.substring(1, comma).toInt();
  int angle   = cmd.substring(comma + 1).toInt();
  setJointAngle((uint8_t)channel, angle);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ);
  delay(500);

  releaseAllServos();
  delay(500);
  moveAllToNeutral();

  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    processCommand(cmd);
  }
}
