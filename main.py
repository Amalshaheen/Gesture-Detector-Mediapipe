import os
import time
import math
import json
import threading
import urllib.parse
import urllib.request
import urllib.error
import serial
import cv2
import mediapipe as mp


def load_env_file(env_path):
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            # Keep exported shell vars as higher priority than file values.
            if key and key not in os.environ:
                os.environ[key] = value

# MediaPipe Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_env_file(os.path.join(BASE_DIR, ".env"))
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
COMMAND_COOLDOWN = 1
camera_index = 4  # Default camera index (0 for built-in webcam)

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

# Telegram Emergency Alert Settings
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
STARTUP_ALERT_ENABLED = os.getenv("STARTUP_ALERT_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
EMERGENCY_ALERT_COOLDOWN = 15  # seconds
last_emergency_alert_time = 0
emergency_latched = False
resolved_telegram_chat_id = TELEGRAM_CHAT_ID
chat_id_lookup_lock = threading.Lock()
last_chat_id_lookup_time = 0
CHAT_ID_LOOKUP_RETRY_SECONDS = 10
telegram_conflict_hint_printed = False

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

    # Only send if the command is DIFFERENT from the last one
    if command_char != last_sent_command:
        # Add a tiny 0.1s debounce limit to prevent noisy fluttering
        if (current_time - last_command_time) >= 0.1:
            if bluetooth_serial and bluetooth_serial.is_open:
                try:
                    bluetooth_serial.write(command_char.encode('utf-8'))
                    last_sent_command = command_char
                    last_command_time = current_time
                except Exception as e:
                    print(f"✗ Send error: {e}")
            else:
                print(f"→ [DEMO] Sent: {command_key} ({command_char})")
                last_sent_command = command_char
                last_command_time = current_time


def send_telegram_message(message_text):
    chat_id = get_telegram_chat_id()
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return

    try:
        endpoint = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": message_text
        }).encode("utf-8")

        request = urllib.request.Request(endpoint, data=payload, method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
        print("✓ Telegram alert sent")
    except Exception as e:
        print(f"✗ Telegram alert failed: {e}")


def get_telegram_chat_id():
    global resolved_telegram_chat_id, last_chat_id_lookup_time, telegram_conflict_hint_printed

    if resolved_telegram_chat_id:
        return resolved_telegram_chat_id

    if not TELEGRAM_BOT_TOKEN:
        now = time.time()
        if (now - last_chat_id_lookup_time) < CHAT_ID_LOOKUP_RETRY_SECONDS:
            return ""

        # Telegram getUpdates only allows one active long-poll request.
        # Guard this lookup so async alert threads do not race each other.
        with chat_id_lookup_lock:
            now = time.time()
            if resolved_telegram_chat_id:
                return resolved_telegram_chat_id
            if (now - last_chat_id_lookup_time) < CHAT_ID_LOOKUP_RETRY_SECONDS:
                return ""
            last_chat_id_lookup_time = now

            try:
                endpoint = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
                with urllib.request.urlopen(endpoint, timeout=5) as response:
                    data = json.loads(response.read().decode("utf-8"))

                results = data.get("result", [])
                if not results:
                    print("! Telegram chat_id not found. Message your bot once, then retry.")
                    return ""

                last_update = results[-1]
                message = last_update.get("message", {})
                chat = message.get("chat", {})
                chat_id = chat.get("id")
                if chat_id is None:
                    return ""

                resolved_telegram_chat_id = str(chat_id)
                print(f"✓ Telegram chat_id detected: {resolved_telegram_chat_id}")
                return resolved_telegram_chat_id
            except urllib.error.HTTPError as e:
                if e.code == 409 and not telegram_conflict_hint_printed:
                    telegram_conflict_hint_printed = True
                    print("! Telegram 409 conflict: disable webhook or set TELEGRAM_CHAT_ID in .env directly.")
                else:
                    print(f"✗ Could not resolve Telegram chat_id: HTTP {e.code}")
                return ""
            except Exception as e:
                print(f"✗ Could not resolve Telegram chat_id: {e}")
                return ""


def send_telegram_message_async(message_text):
    thread = threading.Thread(target=send_telegram_message, args=(message_text,), daemon=True)
    thread.start()


def handle_emergency_alerts(current_command):
    global last_emergency_alert_time, emergency_latched

    now = time.time()

    if current_command == "EMERGENCY":
        # Latch the emergency state and alert once per cooldown window.
        if (not emergency_latched) or (now - last_emergency_alert_time >= EMERGENCY_ALERT_COOLDOWN):
            emergency_latched = True
            last_emergency_alert_time = now
            alert_time = time.strftime("%Y-%m-%d %H:%M:%S")
            send_telegram_message_async(
                f"🚨 EMERGENCY gesture detected!\nTime: {alert_time}"
            )
    elif emergency_latched:
        emergency_latched = False
        clear_time = time.strftime("%Y-%m-%d %H:%M:%S")
        send_telegram_message_async(
            f"✅ Emergency gesture cleared.\nTime: {clear_time}"
        )


def send_startup_notification():
    if not STARTUP_ALERT_ENABLED:
        return

    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    send_telegram_message_async(
        f"🟢 Gesture emergency system is online.\nTime: {start_time}"
    )

def print_result(result, output_image, timestamp_ms: int):
    global latest_result
    latest_result = result

def detect_joystick_gesture(landmarks):
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    index_tip = landmarks[8]
    middle_mcp = landmarks[9]
    middle_tip = landmarks[12]
    ring_mcp = landmarks[13]
    ring_tip = landmarks[16]
    pinky_mcp = landmarks[17]
    pinky_tip = landmarks[20]

    # Base scale: Distance from wrist to index knuckle
    mag_base = math.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y)
    if mag_base == 0: return "STOP"

    # Finger extensions relative to base scale
    mag_index = math.hypot(index_tip.x - index_mcp.x, index_tip.y - index_mcp.y)
    mag_middle = math.hypot(middle_tip.x - middle_mcp.x, middle_tip.y - middle_mcp.y)
    mag_ring = math.hypot(ring_tip.x - ring_mcp.x, ring_tip.y - ring_mcp.y)
    mag_pinky = math.hypot(pinky_tip.x - pinky_mcp.x, pinky_tip.y - pinky_mcp.y)
    index_ratio = mag_index / mag_base
    middle_ratio = mag_middle / mag_base
    ring_ratio = mag_ring / mag_base
    pinky_ratio = mag_pinky / mag_base

    # 1. EMERGENCY: Flat hand (Pinky and Middle extended outwards)
    if middle_ratio > 0.8 and pinky_ratio > 0.8:
        return "EMERGENCY"

    # 2. BACKWARD: Two-finger gesture (index + middle up, ring + pinky down)
    if index_ratio > 0.65 and middle_ratio > 0.65 and ring_ratio < 0.55 and pinky_ratio < 0.55:
        return "BACKWARD"

    # 3. HORN: Fist (all non-thumb fingers curled)
    if index_ratio < 0.55 and middle_ratio < 0.55 and ring_ratio < 0.55 and pinky_ratio < 0.55:
        return "HORN"

    # 4. STOP: Index finger resting/curled
    if index_ratio < 0.65:
        return "STOP"

    # 5. DIRECTIONAL: Absolute angle of index finger in camera frame
    dx = index_tip.x - index_mcp.x
    dy = index_tip.y - index_mcp.y
    angle = math.degrees(math.atan2(dy, dx))

    # Absolute Camera frame angles (Y is down in OpenCV):
    # Straight Up = -90, Left = -180/180, Right = 0, Down = 90
    print(f"Angle: {angle:.1f}°, Index Ratio: {index_ratio:.2f}")
    if -115 <= angle <= -75:
        return "FORWARD"
    elif -75 < angle <= 20:
        return "RIGHT"
    elif angle < -115 or angle > 135:
        return "LEFT"
    else:
        return "STOP"


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

connect_bluetooth()
send_startup_notification()

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(camera_index)
    
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
            for idx in [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]:
                lm = hand_landmarks[idx]
                cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

            current_command = detect_joystick_gesture(hand_landmarks)
            send_command(current_command)

        handle_emergency_alerts(current_command)

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