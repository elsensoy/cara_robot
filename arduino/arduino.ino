const int motorPin = 3; 
const int pawButton = 2;

bool motorState = false;
bool lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

void setup() {
  pinMode(motorPin, OUTPUT);
  pinMode(pawButton, INPUT_PULLUP);
  
  // Note: If using /dev/ttyAMA0 (GPIO pins), ensure 
  // you aren't using the Serial console for Linux!
  Serial.begin(9600); 
  Serial.println("SYSTEM_READY");
}

void loop() {
  checkButton();
  readSerial();
}

void checkButton() {
  bool currentButtonState = digitalRead(pawButton);
  if ((lastButtonState == HIGH) && (currentButtonState == LOW)) {
    if (millis() - lastDebounceTime > debounceDelay) {
      motorState = !motorState;
      digitalWrite(motorPin, motorState ? HIGH : LOW);
      Serial.print("EVENT:BUTTON_TOGGLE:");
      Serial.println(motorState);
      lastDebounceTime = millis();
    }
  }
  lastButtonState = currentButtonState;
}

void readSerial() {
  if (Serial.available() > 0) {
    // Read until the newline character \n sent by Python
    String input = Serial.readStringUntil('\n');
    input.trim();

    // Logic to parse "HEAD:75,BLINK:1"
    // For simplicity, let's look for keywords:
    if (input.indexOf("HEAD:") >= 0) {
      int headPos = extractValue(input, "HEAD:");
      Serial.print("ACTION:MOVING_HEAD_TO:");
      Serial.println(headPos);
      // analogWrite(headServoPin, headPos);  
    }

    if (input.indexOf("BLINK:1") >= 0) {
      Serial.println("ACTION:BLINKING");
      digitalWrite(motorPin, HIGH); 
      delay(200);
      digitalWrite(motorPin, LOW);
    }
  }
}

// Helper function to find the number after a label
int extractValue(String data, String label) {
  int pos = data.indexOf(label);
  if (pos == -1) return -1;
  int start = pos + label.length();
  int end = data.indexOf(',', start);
  if (end == -1) end = data.length();
  return data.substring(start, end).toInt();
}
