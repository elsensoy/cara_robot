const int MOTOR_PIN = 3;
const int BLINK_PIN = 4;

void setup() {
  pinMode(MOTOR_PIN, OUTPUT);
  pinMode(BLINK_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);
  digitalWrite(BLINK_PIN, LOW);

  Serial.begin(9600);
  Serial.println("Arduino ready");
}

void loop() {
  if (Serial.available() > 0) {
    char emotion = Serial.read();

    if (emotion == 'H') {
      Serial.println("Happy received → move head");
      digitalWrite(MOTOR_PIN, HIGH);
      delay(500);
      digitalWrite(MOTOR_PIN, LOW);
    } 
    else if (emotion == 'B') {
      Serial.println("Sad received → blink");
      digitalWrite(BLINK_PIN, HIGH);
      delay(300);
      digitalWrite(BLINK_PIN, LOW);
    } 
    else {
      Serial.println("Neutral or unknown → do nothing");
    }
  }
}

