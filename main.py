import os
import time
import math
import json
import threading
from collections import deque
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
camera_index = 1  # Default camera index (0 for built-in webcam)
COMMAND_HISTORY = deque(maxlen=8)
last_stable_command = "STOP"
last_stable_time = 0.0
NON_STOP_MIN_FRAMES = 3
EMERGENCY_MIN_FRAMES = 6
COMMAND_HOLD_SECONDS = 0.25
orientation_sign = None
MIN_GESTURE_CONFIDENCE = 0.56
MIN_GESTURE_MARGIN = 0.06
PALM_ORIENTATION_MIN_CONFIDENCE = 0.12

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


def _parse_chat_ids(raw_value):
    chat_ids = set()
    normalized = str(raw_value).replace(";", ",")
    for token in normalized.split(","):
        chat_id = token.strip()
        if chat_id:
            chat_ids.add(chat_id)
    return chat_ids


configured_telegram_chat_ids = _parse_chat_ids(TELEGRAM_CHAT_ID)
resolved_telegram_chat_ids = set(configured_telegram_chat_ids)
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
    chat_ids = get_telegram_chat_ids()
    if not TELEGRAM_BOT_TOKEN or not chat_ids:
        return

    endpoint = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    sent_count = 0

    for chat_id in chat_ids:
        try:
            payload = urllib.parse.urlencode({
                "chat_id": chat_id,
                "text": message_text
            }).encode("utf-8")

            request = urllib.request.Request(endpoint, data=payload, method="POST")
            with urllib.request.urlopen(request, timeout=5) as response:
                response.read()
            sent_count += 1
        except Exception as e:
            print(f"✗ Telegram alert failed for chat {chat_id}: {e}")

    if sent_count:
        print(f"✓ Telegram alert sent to {sent_count}/{len(chat_ids)} chats")


def _collect_chat_ids_from_update(update):
    chat_ids = set()

    for key in ["message", "edited_message", "channel_post", "edited_channel_post"]:
        payload = update.get(key, {})
        chat = payload.get("chat", {})
        chat_id = chat.get("id")
        if chat_id is not None:
            chat_ids.add(str(chat_id))

    for key in ["my_chat_member", "chat_member"]:
        payload = update.get(key, {})
        chat = payload.get("chat", {})
        chat_id = chat.get("id")
        if chat_id is not None:
            chat_ids.add(str(chat_id))

    callback_query = update.get("callback_query", {})
    callback_message = callback_query.get("message", {})
    callback_chat = callback_message.get("chat", {})
    callback_chat_id = callback_chat.get("id")
    if callback_chat_id is not None:
        chat_ids.add(str(callback_chat_id))

    return chat_ids


def get_telegram_chat_ids():
    global last_chat_id_lookup_time, telegram_conflict_hint_printed, resolved_telegram_chat_ids

    known_chat_ids = set(resolved_telegram_chat_ids)

    if not TELEGRAM_BOT_TOKEN:
        return sorted(known_chat_ids)

    now = time.time()
    if (now - last_chat_id_lookup_time) < CHAT_ID_LOOKUP_RETRY_SECONDS:
        return sorted(known_chat_ids)

    # Telegram getUpdates only allows one active long-poll request.
    # Guard this lookup so async alert threads do not race each other.
    with chat_id_lookup_lock:
        now = time.time()
        if (now - last_chat_id_lookup_time) < CHAT_ID_LOOKUP_RETRY_SECONDS:
            return sorted(resolved_telegram_chat_ids)

        last_chat_id_lookup_time = now

        try:
            endpoint = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))

            discovered_chat_ids = set()
            for update in data.get("result", []):
                discovered_chat_ids.update(_collect_chat_ids_from_update(update))

            if discovered_chat_ids:
                before_count = len(resolved_telegram_chat_ids)
                resolved_telegram_chat_ids.update(discovered_chat_ids)
                new_count = len(resolved_telegram_chat_ids)
                if new_count > before_count:
                    print(f"✓ Telegram chats discovered: +{new_count - before_count} (total {new_count})")
            elif not resolved_telegram_chat_ids:
                print("! Telegram chat IDs not found. Ask each user to message the bot once.")

            return sorted(resolved_telegram_chat_ids)
        except urllib.error.HTTPError as e:
            if e.code == 409 and not telegram_conflict_hint_printed:
                telegram_conflict_hint_printed = True
                print("! Telegram 409 conflict: disable webhook or set TELEGRAM_CHAT_ID in .env directly.")
            else:
                print(f"✗ Could not resolve Telegram chat IDs: HTTP {e.code}")
            return sorted(resolved_telegram_chat_ids)
        except Exception as e:
            print(f"✗ Could not resolve Telegram chat IDs: {e}")
            return sorted(resolved_telegram_chat_ids)


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


def _distance_2d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _joint_angle_deg(a, b, c):
    abx, aby = a.x - b.x, a.y - b.y
    cbx, cby = c.x - b.x, c.y - b.y
    ab_mag = math.hypot(abx, aby)
    cb_mag = math.hypot(cbx, cby)
    if ab_mag == 0 or cb_mag == 0:
        return 0.0

    cos_theta = (abx * cbx + aby * cby) / (ab_mag * cb_mag)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def _finger_extension_score(landmarks, mcp_idx, pip_idx, dip_idx, tip_idx, palm_center):
    # Blend straightness and fingertip reach to reduce one-metric misclassifications.
    pip_angle = _joint_angle_deg(landmarks[mcp_idx], landmarks[pip_idx], landmarks[dip_idx])
    dip_angle = _joint_angle_deg(landmarks[pip_idx], landmarks[dip_idx], landmarks[tip_idx])
    reach_ratio = _distance_2d(landmarks[tip_idx], palm_center) / max(_distance_2d(landmarks[mcp_idx], palm_center), 1e-6)

    straight_score = 0.5 * _clamp((pip_angle - 115.0) / 55.0) + 0.5 * _clamp((dip_angle - 115.0) / 55.0)
    reach_score = _clamp((reach_ratio - 0.95) / 0.60)
    return 0.65 * straight_score + 0.35 * reach_score


def _thumb_extension_score(landmarks, palm_center):
    # Thumb has a unique kinematic chain; score it independently.
    thumb_mcp_angle = _joint_angle_deg(landmarks[1], landmarks[2], landmarks[3])
    thumb_ip_angle = _joint_angle_deg(landmarks[2], landmarks[3], landmarks[4])
    thumb_reach_ratio = _distance_2d(landmarks[4], palm_center) / max(_distance_2d(landmarks[2], palm_center), 1e-6)

    straight_score = 0.5 * _clamp((thumb_mcp_angle - 110.0) / 60.0) + 0.5 * _clamp((thumb_ip_angle - 110.0) / 60.0)
    reach_score = _clamp((thumb_reach_ratio - 0.90) / 0.55)
    return 0.65 * straight_score + 0.35 * reach_score



def _raw_direction_value(landmarks, palm_scale, fingers):
    finger_map = {
        "thumb": (4, 2),
        "index": (8, 5),
        "middle": (12, 9),
        "ring": (16, 13),
        "little": (20, 17),
    }
    values = []
    for finger_name in fingers:
        tip_idx, mcp_idx = finger_map[finger_name]
        values.append((landmarks[tip_idx].y - landmarks[mcp_idx].y) / max(palm_scale, 1e-6))
    return sum(values) / max(len(values), 1)


def _palm_orientation_raw_value(landmarks, handedness_label):
    # Estimate palm-facing direction from 3D palm plane normal.
    # After handedness compensation, sign is stabilized and later calibrated.
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    little_mcp = landmarks[17]

    v1x, v1y, v1z = index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z
    v2x, v2y, v2z = little_mcp.x - wrist.x, little_mcp.y - wrist.y, little_mcp.z - wrist.z

    nx = v1y * v2z - v1z * v2y
    ny = v1z * v2x - v1x * v2z
    nz = v1x * v2y - v1y * v2x

    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag <= 1e-8:
        return 0.0

    nz_norm = nz / mag
    hand_factor = -1.0 if str(handedness_label).upper() == "LEFT" else 1.0
    return nz_norm * hand_factor


def _calibrate_palm_orientation_sign(finger_scores, palm_raw_value):
    global orientation_sign

    if orientation_sign in (-1, 1):
        return

    # Calibrate using confident full-open palm (user default STOP posture = palm-down).
    full_open = all(finger_scores[f] > 0.72 for f in ["thumb", "index", "middle", "ring", "little"])
    if not full_open:
        return

    if abs(palm_raw_value) < PALM_ORIENTATION_MIN_CONFIDENCE:
        return

    orientation_sign = 1 if palm_raw_value > 0 else -1
    print(f"✓ Palm orientation calibrated (DOWN sign={orientation_sign})")


def _palm_orientation_quality(landmarks, handedness_label, expected_orientation):
    palm_raw = _palm_orientation_raw_value(landmarks, handedness_label)
    if orientation_sign in (-1, 1):
        palm_raw *= orientation_sign

    if expected_orientation == "DOWN":
        return _clamp((palm_raw - 0.02) / 0.20)
    return _clamp((-palm_raw - 0.02) / 0.20)


def calibrate_orientation_sign(finger_scores, landmarks, palm_scale):
    global orientation_sign

    if orientation_sign in (-1, 1):
        return

    # Calibrate only with a confident full-open hand to avoid noisy sign flips.
    full_open = all(finger_scores[f] > 0.72 for f in ["thumb", "index", "middle", "ring", "little"])
    if not full_open:
        return

    direction_value = _raw_direction_value(
        landmarks,
        palm_scale,
        ["thumb", "index", "middle", "ring", "little"],
    )
    if abs(direction_value) < 0.12:
        return

    orientation_sign = 1 if direction_value > 0 else -1
    print(f"✓ Orientation calibrated (DOWN sign={orientation_sign})")


def _gesture_score(finger_scores, req_ext, req_fold):
    ext_quality = sum(finger_scores[f] for f in req_ext) / max(len(req_ext), 1)
    fold_quality = sum((1.0 - finger_scores[f]) for f in req_fold) / max(len(req_fold), 1) if req_fold else 1.0
    return ext_quality, fold_quality


def reset_command_stabilizer():
    global last_stable_command, last_stable_time
    COMMAND_HISTORY.clear()
    last_stable_command = "STOP"
    last_stable_time = time.time()


def stabilize_command(raw_command):
    global last_stable_command, last_stable_time

    now = time.time()

    # Keep previous stable command briefly when classifier is uncertain.
    if raw_command == "UNKNOWN":
        if last_stable_command != "STOP" and (now - last_stable_time) <= COMMAND_HOLD_SECONDS:
            return last_stable_command
        return "STOP"

    COMMAND_HISTORY.append(raw_command)

    counts = {}
    for cmd in COMMAND_HISTORY:
        counts[cmd] = counts.get(cmd, 0) + 1

    candidate, candidate_count = max(counts.items(), key=lambda item: item[1])

    if candidate == "EMERGENCY":
        if candidate_count >= EMERGENCY_MIN_FRAMES:
            last_stable_command = "EMERGENCY"
            last_stable_time = now
            return "EMERGENCY"
        return "STOP"

    if candidate != "STOP" and candidate_count >= NON_STOP_MIN_FRAMES:
        last_stable_command = candidate
        last_stable_time = now
        return candidate

    if last_stable_command != "STOP" and (now - last_stable_time) <= COMMAND_HOLD_SECONDS:
        return last_stable_command

    return "STOP"


def detect_joystick_gesture(landmarks, handedness_label="Right"):
    palm_center = type("Point", (), {
        "x": (landmarks[0].x + landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 5.0,
        "y": (landmarks[0].y + landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 5.0,
    })()

    finger_scores = {
        "thumb": _thumb_extension_score(landmarks, palm_center),
        "index": _finger_extension_score(landmarks, 5, 6, 7, 8, palm_center),
        "middle": _finger_extension_score(landmarks, 9, 10, 11, 12, palm_center),
        "ring": _finger_extension_score(landmarks, 13, 14, 15, 16, palm_center),
        "little": _finger_extension_score(landmarks, 17, 18, 19, 20, palm_center),
    }

    palm_raw = _palm_orientation_raw_value(landmarks, handedness_label)
    _calibrate_palm_orientation_sign(finger_scores, palm_raw)

    if orientation_sign not in (-1, 1):
        return "UNKNOWN"

    gesture_specs = [
        {
            "name": "STOP",
            "targets": {"thumb": 1.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "little": 1.0},
            "orientation": "DOWN",
            "shape_weight": 0.70,
            "orientation_weight": 0.30,
        },
        {
            "name": "BACKWARD",
            "targets": {"thumb": 1.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "little": 1.0},
            "orientation": "UP",
            "shape_weight": 0.70,
            "orientation_weight": 0.30,
        },
        {
            "name": "EMERGENCY",
            "targets": {"thumb": 0.0, "index": 1.0, "middle": 1.0, "ring": 0.0, "little": 0.0},
            "orientation": "UP",
            "shape_weight": 0.56,
            "orientation_weight": 0.30,
            "split_weight": 0.14,
        },
        {
            "name": "HORN",
            "targets": {"thumb": 0.0, "index": 1.0, "middle": 1.0, "ring": 0.0, "little": 0.0},
            "orientation": "DOWN",
            "shape_weight": 0.56,
            "orientation_weight": 0.30,
            "split_weight": 0.14,
        },
        {
            "name": "FORWARD",
            "targets": {"thumb": 0.0, "index": 1.0, "middle": 0.0, "ring": 0.0, "little": 0.0},
            "orientation": "DOWN",
            "shape_weight": 0.62,
            "orientation_weight": 0.24,
            "split_weight": 0.14,
        },
        {
            "name": "LEFT",
            "targets": {"thumb": 1.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "little": 0.0},
            "orientation": "DOWN",
            "shape_weight": 0.62,
            "orientation_weight": 0.24,
            "split_weight": 0.14,
        },
        {
            "name": "RIGHT",
            "targets": {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "little": 1.0},
            "orientation": "DOWN",
            "shape_weight": 0.62,
            "orientation_weight": 0.24,
            "split_weight": 0.14,
        },
    ]

    scored_gestures = []

    for spec in gesture_specs:
        targets = spec["targets"]
        finger_error = sum(abs(finger_scores[f] - targets[f]) for f in targets) / len(targets)
        shape_quality = 1.0 - finger_error
        orientation_quality = _palm_orientation_quality(landmarks, handedness_label, spec["orientation"])
        total_score = spec["shape_weight"] * shape_quality + spec["orientation_weight"] * orientation_quality

        if spec["name"] in {"HORN", "EMERGENCY"}:
            pair_avg = 0.5 * (finger_scores["index"] + finger_scores["middle"])
            others_avg = (finger_scores["thumb"] + finger_scores["ring"] + finger_scores["little"]) / 3.0
            split_quality = _clamp((pair_avg - others_avg + 0.12) / 0.95)
            pair_balance = 1.0 - abs(finger_scores["index"] - finger_scores["middle"])
            thumb_target = targets["thumb"]
            thumb_match = 1.0 - abs(finger_scores["thumb"] - thumb_target)
            total_score += spec["split_weight"] * (0.45 * split_quality + 0.30 * pair_balance + 0.25 * thumb_match)

        if spec["name"] in {"FORWARD", "LEFT", "RIGHT"}:
            if spec["name"] == "FORWARD":
                active = finger_scores["index"]
                passive = (finger_scores["thumb"] + finger_scores["middle"] + finger_scores["ring"] + finger_scores["little"]) / 4.0
            elif spec["name"] == "LEFT":
                active = finger_scores["thumb"]
                passive = (finger_scores["index"] + finger_scores["middle"] + finger_scores["ring"] + finger_scores["little"]) / 4.0
            else:
                active = finger_scores["little"]
                passive = (finger_scores["thumb"] + finger_scores["index"] + finger_scores["middle"] + finger_scores["ring"]) / 4.0

            split_quality = _clamp((active - passive + 0.15) / 1.0)
            total_score += spec["split_weight"] * split_quality

        scored_gestures.append((spec["name"], total_score, shape_quality, orientation_quality))

    scored_gestures.sort(key=lambda item: item[1], reverse=True)
    best_name, best_score, best_shape, best_orientation = scored_gestures[0]
    second_score = scored_gestures[1][1] if len(scored_gestures) > 1 else 0.0

    if best_score < MIN_GESTURE_CONFIDENCE:
        return "UNKNOWN"

    if (best_score - second_score) < MIN_GESTURE_MARGIN:
        return "UNKNOWN"

    # Reject orientation-mismatched matches between same-shape gestures.
    if best_orientation < 0.45:
        return "UNKNOWN"

    # Avoid STOP swallowing uncertain shapes; STOP should look like a clear open hand.
    if best_name == "STOP" and best_shape < 0.72:
        return "UNKNOWN"

    return best_name


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

connect_bluetooth()
send_startup_notification()

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(camera_index)
    reset_command_stabilizer()
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            # Safety fallback: if camera frame read fails, force STOP.
            reset_command_stabilizer()
            send_command("STOP")
            handle_emergency_alerts("STOP")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        landmarker.detect_async(mp_image, int(time.time() * 1000))
        
        current_command = "STOP"

        if latest_result and latest_result.hand_landmarks:
            hand_landmarks = latest_result.hand_landmarks[0]
            handedness_label = "Right"
            if latest_result.handedness and len(latest_result.handedness) > 0 and len(latest_result.handedness[0]) > 0:
                handedness_label = latest_result.handedness[0][0].category_name
            
            # Draw primary landmarks for visual feedback
            h, w = image.shape[:2]
            for idx in [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]:
                lm = hand_landmarks[idx]
                cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

            raw_command = detect_joystick_gesture(hand_landmarks, handedness_label)
            current_command = stabilize_command(raw_command)
            send_command(current_command)
        else:
            # Safety rule: no tracked hand means immediate STOP command.
            reset_command_stabilizer()
            send_command("STOP")
            current_command = "STOP"

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