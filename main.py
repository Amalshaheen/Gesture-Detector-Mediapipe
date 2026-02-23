import os
import time
import math
import serial
import serial.tools.list_ports

import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'hand_landmarker.task')


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
latest_result = None
current_gesture = None
last_sent_command = None
last_command_time = 0
COMMAND_COOLDOWN = 0.5  # Send command every 0.5 seconds to avoid spam

# Gesture to wheelchair command mapping
GESTURE_COMMANDS = {
    "OPEN PALM": "S",      # Stop - 5 fingers extended
    "ROCK": "F",           # Forward - index + pinky extended
    "THREE": "R",          # Right - thumb + index + middle extended
    "PEACE": "R",          # Right - index + middle extended (same as three)
    "L SHAPE": "L",        # Left - thumb + index extended (L shape)
    "FIST": "B",           # Back - all fingers closed
}

# ESP32 Bluetooth connection settings
# --- CHANGE THIS TO YOUR ACTUAL PORT ---
# Windows example: 'COM9'
# Mac example: '/dev/cu.ESP32_LED_Control-ESP32SPP'
# Linux example: '/dev/rfcomm0'
COM_PORT = 'COM9'
BAUD_RATE = 115200
bluetooth_serial = None

def list_available_ports():
    """List all available serial ports."""
    ports = serial.tools.list_ports.comports()
    if ports:
        print("Available serial ports:")
        for port in ports:
            print(f"  - {port.device}: {port.description}")
    else:
        print("No serial ports found.")
    return ports

def connect_bluetooth():
    """Connect to ESP32 via Bluetooth Serial."""
    global bluetooth_serial
    try:
        print(f"Attempting to connect to {COM_PORT}...")
        bluetooth_serial = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(0.5)  # Give time for connection to stabilize
        print(f"✓ Successfully connected to ESP32 on {COM_PORT}!")
        return True
    except serial.SerialException as e:
        print(f"✗ Bluetooth connection failed: {e}")
        print(f"  Could not open port {COM_PORT}")
        print("  Tips:")
        print("  1. Check the COM_PORT variable in the script")
        print("  2. Ensure ESP32 is paired with your laptop")
        print("  3. Check available ports below:")
        list_available_ports()
        print("  Continuing in demo mode (no commands will be sent)")
        bluetooth_serial = None
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        print("  Continuing in demo mode")
        bluetooth_serial = None
        return False

def send_command(command):
    """Send command to ESP32 via Bluetooth Serial."""
    global last_sent_command, last_command_time
    
    current_time = time.time()
    
    # Debounce: only send if command changed or cooldown elapsed
    if command == last_sent_command and (current_time - last_command_time) < COMMAND_COOLDOWN:
        return
    
    if bluetooth_serial and bluetooth_serial.is_open:
        try:
            # Send command to ESP32
            bluetooth_serial.write(command.encode('utf-8'))
            print(f"→ Sent: {command}")
            
            # Wait briefly and check for response
            time.sleep(0.05)
            if bluetooth_serial.in_waiting > 0:
                response = bluetooth_serial.readline().decode('utf-8').strip()
                print(f"  ESP32: {response}")
            
            last_sent_command = command
            last_command_time = current_time
        except serial.SerialException as e:
            print(f"✗ Serial error: {e}")
            print("  Connection may be lost. Continuing in demo mode.")
        except Exception as e:
            print(f"✗ Failed to send command: {e}")
    else:
        # Demo mode - just print
        if command != last_sent_command or (current_time - last_command_time) >= COMMAND_COOLDOWN:
            print(f"→ [DEMO] Command: {command}")
            last_sent_command = command
            last_command_time = current_time


def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


def get_distance(point1, point2):
    """Calculate Euclidean distance between two landmarks."""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)


def is_finger_extended(landmarks, finger_tip_idx, finger_pip_idx, finger_mcp_idx):
    """Check if a finger is extended based on landmark positions."""
    tip = landmarks[finger_tip_idx]
    pip = landmarks[finger_pip_idx]
    mcp = landmarks[finger_mcp_idx]
    
    # Finger is extended if tip is farther from wrist than pip
    wrist = landmarks[0]
    tip_to_wrist = get_distance(tip, wrist)
    pip_to_wrist = get_distance(pip, wrist)
    
    return tip_to_wrist > pip_to_wrist


def is_thumb_extended(landmarks):
    """Check if thumb is extended."""
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    wrist = landmarks[0]
    
    # Thumb is extended if tip is farther from wrist than IP joint
    tip_to_wrist = get_distance(thumb_tip, wrist)
    ip_to_wrist = get_distance(thumb_ip, wrist)
    
    return tip_to_wrist > ip_to_wrist


def detect_gesture(hand_landmarks):
    """Detect gesture based on hand landmarks."""
    landmarks = hand_landmarks
    
    # Check each finger's state
    thumb_extended = is_thumb_extended(landmarks)
    index_extended = is_finger_extended(landmarks, 8, 6, 5)
    middle_extended = is_finger_extended(landmarks, 12, 10, 9)
    ring_extended = is_finger_extended(landmarks, 16, 14, 13)
    pinky_extended = is_finger_extended(landmarks, 20, 18, 17)
    
    # Count extended fingers
    fingers_up = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
    
    # Detect only the required gestures
    
    # Fist - all fingers closed (BACK)
    if fingers_up == 0:
        return "FIST"
    
    # Open Palm - all fingers extended (STOP)
    if fingers_up == 5:
        return "OPEN PALM"
    
    # Rock/Horn - index and pinky extended (FORWARD)
    if index_extended and not middle_extended and not ring_extended and pinky_extended:
        return "ROCK"
    
    # L Shape - thumb and index extended only (LEFT)
    if thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "L SHAPE"
    
    # Peace - index and middle fingers extended (RIGHT)
    if not thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "PEACE"
    
    # Three - thumb, index, and middle extended (RIGHT)
    if thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "THREE"
    
    # No recognized gesture
    return "UNKNOWN"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# Connect to ESP32 Bluetooth
print("Connecting to ESP32...")
connect_bluetooth()

with HandLandmarker.create_from_options(options) as landmarker:
    # Continuously capture images from the webcam and feed them into the hand landmarker:
    import cv2

    def draw_hand_landmarks(frame, hand_landmarks):
        height, width = frame.shape[:2]
        points = []
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Standard MediaPipe hand connections (21 landmarks).
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]
        for start, end in connections:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (0, 0, 255), 2)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        landmarker.detect_async(mp_image, int(time.time() * 1000))
        image.flags.writeable = True
        
        # Draw landmarks and detect gestures
        if latest_result and latest_result.hand_landmarks:
            gestures = []
            for hand_landmarks in latest_result.hand_landmarks:
                draw_hand_landmarks(image, hand_landmarks)
                gesture = detect_gesture(hand_landmarks)
                gestures.append(gesture)
            
            # Use the first detected hand's gesture for control
            if gestures:
                primary_gesture = gestures[0]
                
                # Send wheelchair command if gesture is mapped
                if primary_gesture in GESTURE_COMMANDS:
                    command = GESTURE_COMMANDS[primary_gesture]
                    send_command(command)
                    
                    # Display gesture and command
                    cv2.putText(image, f"Gesture: {primary_gesture}", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Command: {command}", (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
                else:
                    # Gesture detected but not mapped
                    cv2.putText(image, f"Gesture: {primary_gesture}", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2, cv2.LINE_AA)
                    cv2.putText(image, "No command", (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2, cv2.LINE_AA)
        else:
            # No hand detected
            cv2.putText(image, "No hand detected", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display connection status
        status = "Connected" if (bluetooth_serial and bluetooth_serial.is_open) else "Demo Mode"
        status_color = (0, 255, 0) if (bluetooth_serial and bluetooth_serial.is_open) else (255, 0, 0)
        cv2.putText(image, f"BT: {status}", (10, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
        
        cv2.imshow('Wheelchair Gesture Control', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Close Bluetooth connection
    if bluetooth_serial and bluetooth_serial.is_open:
        try:
            bluetooth_serial.close()
            print("✓ Bluetooth serial connection closed")
        except Exception as e:
            print(f"✗ Error closing connection: {e}")