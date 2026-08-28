import RPi.GPIO as GPIO
import time
import atexit

# --- Configuration ---
# We are using physical pin numbers (GPIO.BOARD mode)
# This matches the pinout diagram.
MOTOR_PIN = 11  # Was Arduino D3
BLINK_PIN = 12  # Was Arduino D2
# ---------------------

def initialize_arduino():
    """
    (Function name kept for compatibility with cara.py)
    
    This function now initializes the Jetson's GPIO pins
    instead of opening a serial port.
    """
    try:
        # Suppress warnings (e.g., "This channel is already in use")
        GPIO.setwarnings(False) 
        
        # Use physical pin numbering (1, 2, 3... 40)
        GPIO.setmode(GPIO.BOARD) 
        
        # Set our pins as outputs and set their initial state to LOW (off)
        GPIO.setup(MOTOR_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(BLINK_PIN, GPIO.OUT, initial=GPIO.LOW)
        
        print(f"SUCCESS: Jetson GPIO pins {MOTOR_PIN} (Motor) and {BLINK_PIN} (Blink) initialized.")
        return True
    
    except Exception as e:
        print(f"ERROR: Could not initialize Jetson GPIO.")
        print(f"Details: {e}")
        print("---")
        print("Did you remember to run this script with 'sudo'?")
        print("Try: sudo python3 cara.py")
        print("---")
        return False

def send_emotion_to_arduino(emotion_string):
    """
    (Function name kept for compatibility with cara.py)
    
    This function now directly controls the GPIO pins,
    replicating the logic that was inside the Arduino.
    """
    try:
        if emotion_string == "happy":
            # Replicate the Arduino's 'H' logic
            print("[GPIO]: Happy received -> move head")
            GPIO.output(MOTOR_PIN, GPIO.HIGH)
            time.sleep(0.5) # This was delay(500)
            GPIO.output(MOTOR_PIN, GPIO.LOW)
        
        elif emotion_string == "sad":
            # Replicate the Arduino's 'B' logic
            print("[GPIO]: Sad received -> blink")
            GPIO.output(BLINK_PIN, GPIO.HIGH)
            time.sleep(0.3) # This was delay(300)
            GPIO.output(BLINK_PIN, GPIO.LOW)
        
        else:
            # Replicate the "else" block (do nothing, ensure pins are off)
            print("[GPIO]: Neutral or unknown -> do nothing")
            GPIO.output(MOTOR_PIN, GPIO.LOW)
            GPIO.output(BLINK_PIN, GPIO.LOW)
    
    except Exception as e:
        print(f"ERROR: Failed to write to Jetson GPIO pins. {e}")

def close_arduino():
    """
    (Function name kept for compatibility with cara.py)
    
    This function now cleans up the GPIO pins on exit.
    """
    print("Shutting down and cleaning up Jetson GPIO pins...")
    GPIO.cleanup()

# Register the cleanup function to be called when the script exits
# This is very important for GPIO!
atexit.register(close_arduino)
