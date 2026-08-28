import Jetson.GPIO as GPIO
import time

# Pin Definitions
motor_pin = 32  # Must be a PWM pin
button_pin = 12 # Pull-up button pin

# Global States
motor_state = False
last_button_state = GPIO.HIGH

def setup():
    GPIO.setmode(GPIO.BOARD)
    # Set up button with internal pull-up resistor
    GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    # Set up motor pin as PWM at 50Hz
    GPIO.setup(motor_pin, GPIO.OUT)
    global pwm
    pwm = GPIO.PWM(motor_pin, 50) 
    pwm.start(0)
    
    print("🧸 Cara Diagnostic Mode Initialized (Jetson Edition)")
    print("--------------------------------------------------")

def loop():
    global motor_state, last_button_state
    
    current_button_state = GPIO.input(button_pin)

    # Detect Button Press (Falling Edge)
    if last_button_state == GPIO.HIGH and current_button_state == GPIO.LOW:
        motor_state = not motor_state
        
        if motor_state:
            print("\n🔄 Motor turning ON")
            # Arduino 200/255 is roughly 78% duty cycle
            pwm.ChangeDutyCycle(78) 
        else:
            print("⛔ Motor turning OFF")
            pwm.ChangeDutyCycle(0)
            
        time.sleep(0.2) # Simple debounce

    last_button_state = current_button_state

try:
    setup()
    while True:
        loop()
        time.sleep(0.01) # Prevent CPU hogging
finally:
    pwm.stop()
    GPIO.cleanup()
