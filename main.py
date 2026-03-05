import os
import time
import math
import serial
import serial.tools.list_ports
import cv2
import mediapipe as mp

# MediaPipe Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'hand_landmarker.task')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global Variables
latest_result = None
last_sent_command = None
last_command_time = 0
COMMAND_COOLDOWN = 0.3

GESTURE_COMMANDS = {
    "FORWARD": "F",
    "BACKWARD": "B",
    "LEFT": "L",
    "RIGHT": "R",
    "STOP": "S",
    "HORN": "H",
    "EMERGENCY": "E"
}

# Bluetooth Settings
COM_PORT = '/dev/rfcomm0'
BAUD_RATE = 115200
bluetooth_serial = None

def connect_bluetooth():
    global bluetooth_serial
    try:
        bluetooth_serial = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(0.5)
        print(f"✓ Connected to ESP32 on {COM_PORT}")
        return True
    except Exception as e:
        print(f"✗ Bluetooth failed: {e}. Running in Demo Mode.")
        bluetooth_serial = None
        return False

def send_command(command_key):
    global last_sent_command, last_command_time
    current_time = time.time()
    
    command_char = GESTURE_COMMANDS.get(command_key, "S")

    if command_char == last_sent_command and (current_time - last_command_time) < COMMAND_COOLDOWN:
        return
    
    if bluetooth_serial and bluetooth_serial.is_open:
        try:
            bluetooth_serial.write(command_char.encode('utf-8'))
            last_sent_command = command_char
            last_command_time = current_time
        except Exception as e:
            print(f"✗ Send error: {e}")
    else:
        if command_char != last_sent_command or (current_time - last_command_time) >= COMMAND_COOLDOWN:
            print(f"→ [DEMO] Sent: {command_key} ({command_char})")
            last_sent_command = command_char
            last_command_time = current_time

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def detect_joystick_gesture(landmarks):
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    index_tip = landmarks[8]
    middle_mcp = landmarks[9]
    middle_tip = landmarks[12]
    pinky_mcp = landmarks[17]
    pinky_tip = landmarks[20]

    # Base scale: Distance from wrist to index knuckle
    mag_base = math.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y)
    if mag_base == 0: return "STOP"

    # Finger extensions relative to base scale
    mag_index = math.hypot(index_tip.x - index_mcp.x, index_tip.y - index_mcp.y)
    mag_middle = math.hypot(middle_tip.x - middle_mcp.x, middle_tip.y - middle_mcp.y)
    mag_pinky = math.hypot(pinky_tip.x - pinky_mcp.x, pinky_tip.y - pinky_mcp.y)

    # 1. EMERGENCY: Flat hand (Pinky and Middle extended outwards)
    if (mag_middle / mag_base) > 0.8 and (mag_pinky / mag_base) > 0.8:
        return "EMERGENCY"

    # 2. HORN: Scissor (Middle extended and far from index tip)
    scissor_dist = math.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y)
    if (mag_middle / mag_base) > 0.5 and (scissor_dist / mag_base) > 0.6:
        return "HORN"

    # 3. STOP: Index finger resting/curled
    index_ratio = mag_index / mag_base
    if index_ratio < 0.65:
        return "STOP"

    # 4. DIRECTIONAL: Absolute angle of index finger in camera frame
    dx = index_tip.x - index_mcp.x
    dy = index_tip.y - index_mcp.y
    angle = math.degrees(math.atan2(dy, dx))

    # Absolute Camera frame angles (Y is down in OpenCV):
    # Straight Up = -90, Left = -180/180, Right = 0, Down = 90
    print(f"Angle: {angle:.1f}°, Index Ratio: {index_ratio:.2f}")
    if -115 <= angle <= -65:
        return "FORWARD"
    elif -65 < angle <= 20:
        return "RIGHT"
    elif angle < -115 or angle > 135:
        return "LEFT"
    else:
        return "BACKWARD"


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

connect_bluetooth()

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(4)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        landmarker.detect_async(mp_image, int(time.time() * 1000))
        
        current_command = "STOP"

        if latest_result and latest_result.hand_landmarks:
            hand_landmarks = latest_result.hand_landmarks[0]
            
            # Draw primary landmarks for visual feedback
            h, w = image.shape[:2]
            for idx in [0, 5, 8, 9, 12, 17, 20]:
                lm = hand_landmarks[idx]
                cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

            current_command = detect_joystick_gesture(hand_landmarks)
            send_command(current_command)

        # UI Overlay
        cv2.putText(image, f"CMD: {current_command}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        cv2.imshow('Wheelchair Control', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    if bluetooth_serial and bluetooth_serial.is_open:
        bluetooth_serial.close()