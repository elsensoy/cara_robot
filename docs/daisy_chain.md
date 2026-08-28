To handle 20 servos, we will need two of these boards daisy-chained together.
The Hardware Architecture

The Jetson Orin Nano will act as the "brain," while the two PCA9685 boards act as the "nervous system."

    Connection: Jetson I2C Pins ->  PCA9685 #1 ->  PCA9685 #2.

    Addressing: Board #1 is usually at address 0x40. Board #2 needs a solder bridge on the "A0" jumper to change its address to 0x41.

    Power: Do not power the servos from the Jetson. Use a dedicated 5V–6V high-current power supply (or a 2S LiPo with a BEC) connected to the terminal blocks on the PCA9685 boards.

20-Servo Pinout Mapping (Research Standard)

Here is how we should organize the wires to keep Cara’s movements logical in code:

Board 1 (Address 0x40): Core Locomotion & Torso | Pin | Joint | Movement | | :--- | :--- | :--- | | 0–2 | Waist | Pitch, Yaw, Roll | | 3–5 | Right Hip | Flexion, Abduction, Rotation | | 6–8 | Left Hip | Flexion, Abduction, Rotation | | 9–11 | Neck | Pitch, Yaw, Roll | | 12–15 | Spare | Extra sensors or cooling fans |

Board 2 (Address 0x41): Upper Body & Expression | Pin | Joint | Movement | | :--- | :--- | :--- | | 0–2 | Right Shoulder | Flexion, Abduction, Rotation | | 3–5 | Left Shoulder | Flexion, Abduction, Rotation | | 6 | Right Ear | 1-DoF Expression | | 7 | Left Ear | 1-DoF Expression | | 8–15 | Expansion | Future hands, eyelids, or tail-less wag |



Pro-Tips for Build

    Common Ground: Ensure the Ground (GND) from the external battery is connected to both the PCA9685 boards and a GND pin on the Jetson Orin Nano. Without a common ground, the PWM signals will be "noisy" and the servos will jitter.

    Cable Management: With 20 servos, the "mid-body" will become a "rat's nest" of wires quickly. Since we are 3D printing the skeleton, ask MeshAI to include "recessed cable channels" to keep the wires tucked away from the moving joints.

    Voltage Sag: When all 20 servos move at once (e.g., during a jump or a wave), they will pull a lot of current. Add a large capacitor (e.g., 1000µF) to the PCA9685 power terminals to prevent the system from resetting.
    
    
. Daisy-Chaining for 20 Servos

As discussed, we need two boards. In the ROS 2 configuration file (bus_config.yaml), we will define the two addresses:

    Board 1 (0x40): Handles "High Energy" joints (Waist, Hips).

    Board 2 (0x41): Handles "Expression" joints (Shoulders, Neck, Ears).

3. I2C Setup on Orin Nano Super

Before running ROS, we must ensure the Orin can see the drivers.

    Grant Permissions: sudo usermod -aG i2c $USER

    Scan for Boards: sudo i2cdetect -y -r 1 (or bus 7, depending on the header).

    Validation: we should see 40 and 41 in the grid output. If don't, check SDA/SCL wiring.
    
    
    
    
    
    
