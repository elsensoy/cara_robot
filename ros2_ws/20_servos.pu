from adafruit_servokit import ServoKit

# Initialize both boards
# Board 1: 16 channels, address 0x40
kit1 = ServoKit(channels=16, address=0x40)
# Board 2: 16 channels, address 0x41
kit2 = ServoKit(channels=16, address=0x41)

# Example: Move Cara's waist pitch and ears
def express_curiosity():
    kit1.servo[0].angle = 110  # Lean waist forward
    kit2.servo[6].angle = 15   # Perking right ear
    kit2.servo[7].angle = 15   # Perking left ear

express_curiosity()
