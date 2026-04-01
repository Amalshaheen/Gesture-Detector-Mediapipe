# Gesture Detector - Complete Code Documentation

## Project Overview

The Gesture Detector is an intelligent gesture recognition system designed to control a wheelchair using hand gestures captured via a camera. It uses Google's MediaPipe library to detect hand landmarks in real-time, processes these landmarks through a sophisticated gesture classification algorithm, and sends control commands to an ESP32 microcontroller via Bluetooth. Additionally, it provides emergency alert functionality through Telegram notifications.

### Key Features
- **Real-time hand gesture detection** using MediaPipe's HandLandmarker model
- **7 distinct gestures** for wheelchair control (Forward, Backward, Left, Right, Stop, Horn, Emergency)
- **Bluetooth communication** with hardware (ESP32 microcontroller)
- **Telegram emergency notifications** with automatic chat ID discovery
- **Robust command stabilization** to prevent noisy or accidental commands
- **Demo mode** that runs without hardware if Bluetooth connection fails
- **Environment-based configuration** via `.env` files

---

## Project Structure

```
Gesture Detector/
├── main.py                  # Main application entry point
├── hand_landmarker.task     # Pre-trained MediaPipe hand detection model
├── hardwarecode.ino         # Arduino/ESP32 firmware for wheelchair control
├── .env                     # Environment configuration (optional)
├── README.md                # Project readme
└── CODE_DOCUMENTATION.md    # This file
```

---

## Core Technologies & Imports

```python
import os, time, math, json, threading, urllib, serial, cv2, mediapipe as mp
```

### External Libraries Used

| Library | Purpose |
|---------|---------|
| **MediaPipe** | Hand landmark detection and gesture recognition |
| **OpenCV (cv2)** | Video capture and real-time image display |
| **PySerial** | Bluetooth communication with ESP32 |
| **Threading** | Asynchronous Telegram notifications |
| **urllib** | HTTP requests for Telegram API |

---

## Configuration System

### Environment Variables

The system loads configuration from a `.env` file in the project directory:

```python
def load_env_file(env_path):
    """Load environment variables from .env file, giving priority to already-exported system variables."""
```

**Supported Environment Variables:**

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `TELEGRAM_BOT_TOKEN` | String | Empty | Telegram bot token for sending alerts |
| `TELEGRAM_CHAT_ID` | String | Empty | Telegram chat ID(s) for notifications (comma or semicolon separated) |
| `STARTUP_ALERT_ENABLED` | Boolean | "1" | Send notification when system starts |

### Global Configuration Constants

```python
# Camera Settings
camera_index = 1                # Default camera (0 = built-in webcam)

# Gesture Recognition Thresholds
MIN_GESTURE_CONFIDENCE = 0.56   # Minimum confidence required to classify a gesture
MIN_GESTURE_MARGIN = 0.06       # Minimum score difference between top 2 gestures
PALM_ORIENTATION_MIN_CONFIDENCE = 0.12

# Command Stabilization (prevent false commands from noisy frames)
NON_STOP_MIN_FRAMES = 3         # Frames required before executing non-stop commands
EMERGENCY_MIN_FRAMES = 6        # Frames required before executing emergency commands
COMMAND_HOLD_SECONDS = 0.25     # Duration to hold previous command during uncertain frames
COMMAND_COOLDOWN = 1            # Minimum time between different commands
COMMAND_HISTORY = deque(maxlen=8)  # History buffer for stabilization

# Bluetooth Settings
COM_PORT = '/dev/rfcomm0'       # Serial port for Bluetooth device
BAUD_RATE = 115200              # Serial communication speed

# Telegram Alert Settings
EMERGENCY_ALERT_COOLDOWN = 15   # Minimum seconds between emergency alerts
STARTUP_ALERT_ENABLED = True    # Send startup notification
```

### Gesture Command Mapping

```python
GESTURE_COMMANDS = {
    "FORWARD": "F",     # Single index finger pointing down
    "BACKWARD": "B",    # Open palm, fingers pointing up
    "LEFT": "L",        # Single thumb extended
    "RIGHT": "R",       # Single little finger extended
    "STOP": "S",        # All fingers extended (open palm)
    "HORN": "H",        # Index and middle fingers down (palm down)
    "EMERGENCY": "E"    # Index and middle fingers down (palm up)
}
```

---

## Gesture Detection System

### Hand Landmarks Overview

The MediaPipe HandLandmarker detects 21 hand landmarks per hand:

```
Landmark Indices:
0   = Wrist
1-4 = Thumb (MCP, PIP, DIP, Tip)
5-8 = Index finger
9-12 = Middle finger
13-16 = Ring finger
17-20 = Little finger
```

### Main Detection Function

```python
def detect_joystick_gesture(landmarks, handedness_label="Right"):
    """
    Analyzes hand landmarks and returns the detected gesture.
    
    Returns:
        str: One of "STOP", "FORWARD", "BACKWARD", "LEFT", "RIGHT", "HORN", "EMERGENCY", or "UNKNOWN"
    """
```

#### Step-by-Step Gesture Detection Process

1. **Calculate Palm Center**: Averages positions of wrist and MCP joints (landmarks 0, 5, 9, 13, 17)

2. **Calculate Finger Extension Scores** (0.0 to 1.0):
   - For each finger, calculates a "straightness" metric (how extended is the finger?)
   - Uses joint angles (MCP-PIP-DIP, PIP-DIP-TIP) to determine straightness
   - Combines with "reach" metric (how far is fingertip from palm center?)
   - Formula: `0.65 × straightness + 0.35 × reach`

3. **Calibrate Palm Orientation**:
   - Detects whether the user's palm is naturally facing up or down
   - Uses the 3D normal vector of the palm plane
   - Sets `orientation_sign` to ±1 based on first full-open hand gesture

4. **Score Each Gesture Template**:
   - Each gesture has a template with target finger positions and expected palm orientation
   - For each gesture, calculates:
     - **Shape Quality**: How well finger states match the target (0.0 to 1.0)
     - **Orientation Quality**: How well palm orientation matches (0.0 to 1.0)
     - **Split Quality** (for complex gestures): How well certain fingers are separated from others
   - **Total Score** = (shape_weight × shape_quality) + (orientation_weight × orientation_quality) + (split_weight × split_quality)

5. **Return Best Match**:
   - Selects gesture with highest score
   - Rejects if score < MIN_GESTURE_CONFIDENCE (0.56)
   - Rejects if top 2 scores are too close (margin < MIN_GESTURE_MARGIN = 0.06)
   - Rejects if palm orientation doesn't match (quality < 0.45)
   - Returns "UNKNOWN" if all checks fail

### Gesture Templates

| Gesture | Finger State | Palm Orientation | Purpose |
|---------|--------------|------------------|---------|
| **STOP** | All extended (1.0) | DOWN | Safe default state |
| **BACKWARD** | All extended (1.0) | UP | Move wheelchair backward |
| **FORWARD** | Index only (1.0) | DOWN | Move wheelchair forward |
| **LEFT** | Thumb only (1.0) | DOWN | Turn wheelchair left |
| **RIGHT** | Little finger only (1.0) | DOWN | Turn wheelchair right |
| **HORN** | Index + Middle (1.0) | DOWN | Sound horn |
| **EMERGENCY** | Index + Middle (1.0) | UP | Emergency stop signal |

### Key Helper Functions

#### Finger Extension Scoring
```python
def _finger_extension_score(landmarks, mcp_idx, pip_idx, dip_idx, tip_idx, palm_center):
    """Score how extended a finger is based on joint angles and reach distance."""
    # Returns 0.0 (folded) to 1.0 (fully extended)
```

#### Joint Angle Calculation
```python
def _joint_angle_deg(a, b, c):
    """Calculate angle at point b between points a and c."""
    # Uses dot product to find angle in degrees
```

#### Direction Scoring
```python
def _direction_score(landmarks, palm_scale, fingers, direction):
    """Determine if fingers are pointing up or down relative to MCPs."""
    # direction: "UP" or "DOWN"
```

#### Palm Orientation Detection
```python
def _palm_orientation_raw_value(landmarks, handedness_label):
    """
    Estimate if palm is facing up or down using 3D cross product.
    Returns normalized z-component of palm normal vector.
    """
```

---

## Command Stabilization System

The gesture detection is inherently noisy due to camera artifacts and hand micro-movements. The stabilization system prevents false commands:

```python
def stabilize_command(raw_command):
    """Convert raw noisy gestures into stable, reliable commands."""
```

### Stabilization Algorithm

1. **Maintain History Buffer**: Keeps last 8 raw gesture detections

2. **Count Gesture Occurrences**:
   - Counts how many times each gesture appears in recent history
   - Finds the most frequent gesture

3. **Apply Gesture-Specific Thresholds**:
   - **EMERGENCY**: Requires 6+ consecutive frames (most strict)
   - **Non-STOP commands**: Require 3+ consecutive frames
   - **STOP**: Always accepted (low barrier)

4. **Hold Command During Uncertainty**:
   - If detector returns "UNKNOWN", holds the previous command for up to 0.25 seconds
   - Smooths out brief flickering between valid commands

5. **Extract and Return Stable Command**

### Safety Rules
- When hand tracking is lost:
  ```python
  reset_command_stabilizer()  # Clear history
  send_command("STOP")        # Force safe state
  ```
- If camera frame capture fails, immediately send STOP

---

## Hardware Communication

### Bluetooth Serial Communication

```python
def connect_bluetooth():
    """Establish connection to ESP32 microcontroller via Bluetooth."""
    # On Linux: /dev/rfcomm0 (created by pairing process)
    # Returns True on success, False if device not found (Demo Mode)
```

#### Demo Mode
If Bluetooth connection fails, the system continues running in "Demo Mode":
- Commands are printed to console instead of sent to hardware
- Full gesture recognition and stabilization still functions
- Useful for testing without hardware

```python
def send_command(command_key):
    """Send command character to ESP32 via Bluetooth serial."""
    # Debounce: only sends if command differs from last one
    # Includes 0.1s debounce window to prevent rapid switching
    # On success: writes to serial
    # On failure: prints "[DEMO] Sent: {command}"
```

### Bluetooth Command Protocol

Single character commands sent as UTF-8:

```
'F' → FORWARD
'B' → BACKWARD
'L' → LEFT
'R' → RIGHT
'S' → STOP
'H' → HORN
'E' → EMERGENCY
```

---

## Telegram Integration

### Emergency Notification System

```python
def handle_emergency_alerts(current_command):
    """Send Telegram alerts when emergency gesture is detected/cleared."""
```

**Behavior:**
- Sends alert immediately on first detection
- Does NOT re-send if already in emergency state
- Respects EMERGENCY_ALERT_COOLDOWN (15 seconds minimum between alerts)
- Sends clear notification when emergency gesture is released
- All alerts sent asynchronously in background threads

### Chat ID Auto-Discovery

Telegram chat IDs can be provided in two ways:

1. **Direct Configuration** (via `.env`):
   ```
   TELEGRAM_CHAT_ID="12345,67890"  # Single or comma/semicolon-separated
   ```

2. **Automatic Discovery** (if no chat ID provided):
   ```python
   def get_telegram_chat_ids():
       """
       Query Telegram API to discover all chats that have messaged the bot.
       Caches result for CHAT_ID_LOOKUP_RETRY_SECONDS (10 seconds).
       """
   ```
   - Calls Telegram's `getUpdates` endpoint
   - Extracts chat IDs from all message types
   - Thread-safe (uses lock to prevent concurrent API requests)
   - Only works if users have messaged the bot at least once

### Telegram Helper Functions

```python
def _collect_chat_ids_from_update(update):
    """Extract chat IDs from a Telegram API update object."""
    # Handles messages, edited messages, channel posts, chat member events

def send_telegram_message_async(message_text):
    """Send message asynchronously to avoid blocking main thread."""
    # Spawns daemon thread running send_telegram_message()

def send_telegram_message(message_text):
    """Send alert to all discovered/configured chat IDs."""
```

---

## Main Application Loop

### Initialization

```python
# Set up MediaPipe HandLandmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

connect_bluetooth()              # Attempt BT connection (continues if fails)
send_startup_notification()      # Send Telegram startup alert (if enabled)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(camera_index)  # Open camera feed
```

### Main Loop Flow

```python
while cap.isOpened():
    # 1. Read frame from camera
    success, image = cap.read()
    
    # 2. Convert BGR to RGB (MediaPipe expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. Create MediaPipe image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # 4. Run hand detection asynchronously
    landmarker.detect_async(mp_image, int(time.time() * 1000))
    
    # 5. Check if hands were detected
    if latest_result and latest_result.hand_landmarks:
        hand_landmarks = latest_result.hand_landmarks[0]
        handedness_label = extract_handedness()  # "Right" or "Left"
        
        # 6. Detect gesture from landmarks
        raw_command = detect_joystick_gesture(hand_landmarks, handedness_label)
        
        # 7. Stabilize to reliable command
        current_command = stabilize_command(raw_command)
        
        # 8. Send to hardware
        send_command(current_command)
        
        # 9. Draw visual feedback (circles on key landmarks)
        draw_landmarks(image, hand_landmarks)
    else:
        # Hand lost: force safe state
        reset_command_stabilizer()
        send_command("STOP")
        current_command = "STOP"
    
    # 10. Send emergency alert if needed
    handle_emergency_alerts(current_command)
    
    # 11. Display command on screen
    cv2.putText(image, f"CMD: {current_command}", ...)
    cv2.imshow('Wheelchair Control', image)
    
    # 12. Exit on ESC key (key code 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if bluetooth_serial and bluetooth_serial.is_open:
    bluetooth_serial.close()
```

### Result Callback

```python
def print_result(result, output_image, timestamp_ms: int):
    """Store latest detection result for main loop to process."""
    global latest_result
    latest_result = result  # Async callback called by MediaPipe
```

---

## Data Flow Diagram

```
Camera Feed
    ↓
OpenCV VideoCapture
    ↓
RGB Conversion
    ↓
MediaPipe HandLandmarker (Async Detection)
    ↓
Hand Landmarks [21 points per hand]
    ↓
detect_joystick_gesture() 
  ├─ Calculate finger extension scores
  ├─ Detect palm orientation
  ├─ Score against gesture templates
  └─ Return best match (or "UNKNOWN")
    ↓
stabilize_command()
  ├─ Maintain command history
  ├─ Apply frame thresholds
  └─ Return filtered command
    ↓
send_command()
  ├─ Bluetooth → ESP32 (if connected)
  └─ Console output (Demo Mode)
    ↓
handle_emergency_alerts()
  └─ Telegram (async notification)
    ↓
Display on Screen
```

---

## Important Deep Dives

### Why Gesture Confidence Thresholds Matter

The system uses scoring rather than hard thresholds because:
- **Eliminates false positives**: Overlapping gestures (e.g., HORN vs EMERGENCY) are clearly distinguished by scores
- **Handles lighting variation**: Score-based approach is more robust to inconsistent lighting
- **Avoids "jumping"**: The MIN_GESTURE_MARGIN prevents switching between similar-looking gestures
- **Orientation matching**: Only accepts gestures where palm orientation is confident (0.45+ quality)

### Palm Orientation Calibration

On startup, the system automatically calibrates palm orientation:
1. Waits for user to show a full-open hand (all 5 fingers extended)
2. Calculates the 3D normal vector of the palm plane using cross product
3. Sets `orientation_sign` to ±1 based on Z-component of normal
4. All subsequent gestures use this calibration

**Why?** Different users may hold the device at different angles, so calibration ensures consistency.

### Stabilization Frame Counting

The system uses a sliding window buffer (deque) to maintain recent gesture history:
- **Prevents flickering**: Multiple detector updates must agree before changing state
- **Emergency precedence**: EMERGENCY requires 6 frames (stricter) to prevent accidental activation
- **STOP is forgiving**: Can transition immediately (1 frame) for safety
- **Hold previous state**: During uncertain frames, holds previous non-STOP command

### Bluetooth Debouncing

The system prevents Bluetooth "flutter" (rapid on-off switching):
```python
# Only sends if:
# 1. Command character differs from last sent
# 2. At least 0.1 seconds have passed since last send
```

This reduces noise on Bluetooth line and prevents overwhelming the ESP32.

---

## Error Handling & Safety

| Scenario | Behavior |
|----------|----------|
| **Camera capture fails** | Send STOP, reset stabilizer, continue loop |
| **Bluetooth not available** | Print message, enter Demo Mode, continue |
| **Hand tracking lost** | Send STOP immediately, reset stabilizer |
| **Invalid gesture** | Return "UNKNOWN", hold previous command (0.25s max) |
| **Telegram API error** | Print error, continue (emergency system independent) |
| **Telegram webhook conflict** | Hint to disable webhook or set TELEGRAM_CHAT_ID in .env |

---

## Performance Considerations

### Real-time Constraints
- **MediaPipe async detection**: Non-blocking, runs ~30 FPS on modern hardware
- **Gesture classification**: O(7) gesture template comparisons, negligible overhead
- **Stabilization**: O(8) deque operations, constant time
- **Telegram alerts**: Spawned in separate daemon threads, doesn't block UI loop

### Resource Usage
- **Memory**: ~100-200 MB typical (MediaPipe model + OpenCV buffers)
- **CPU**: 15-25% single core at 30 FPS (depends on hardware)
- **Network**: Only for Telegram alerts (~0.5 KB per alert, ~1s latency)

---

## Configuration Examples

### Minimum `.env` (Demo Mode)
```
# No Bluetooth, no Telegram
# System runs fully in Demo Mode
```

### With Bluetooth Only
```
# Requires pre-paired Bluetooth device on /dev/rfcomm0
# Demo mode falls back if device unavailable
```

### Full Configuration
```env
TELEGRAM_BOT_TOKEN=123456789:ABCDefGHijKlmnoPQRstUVwxyz
TELEGRAM_CHAT_ID=987654321
STARTUP_ALERT_ENABLED=1
```

---

## Testing & Debugging

### Visual Feedback
- Screen shows detected command in yellow text: `CMD: FORWARD`
- Green circles drawn on key landmarks for debugging

### Console Output
```
✓ Connected to ESP32 on /dev/rfcomm0          # Bluetooth OK
✗ Bluetooth failed: ... Running in Demo Mode. # Fallback
→ [DEMO] Sent: FORWARD (F)                    # Demo command
✓ Palm orientation calibrated (DOWN sign=1)   # Calibration done
✓ Telegram alert sent to 1/1 chats            # Notification sent
! Telegram chat IDs not found. Ask each user...# No recipients
```

### Debug Tips
1. Check console output during startup for connection status
2. Verify camera works: `v4l2-info /dev/video0`
3. Test Bluetooth pairing: `bluetoothctl show`
4. Verify Telegram token: Try REST API call manually
5. Test gestures with camera feed visible on-screen

---

## Summary

The Gesture Detector is a sophisticated real-time gesture recognition system that combines:
- **Computer Vision**: MediaPipe for reliable hand detection
- **Machine Learning**: Gesture template matching with confidence scoring
- **Signal Processing**: Command stabilization to filter noise
- **Hardware Integration**: Bluetooth communication with microcontroller
- **Notifications**: Telegram for emergency alerts
- **Robustness**: Graceful fallbacks for missing hardware/services

The modular design allows operation in multiple modes (full hardware, Bluetooth only, or complete Demo Mode) without code changes—all determined by `.env` configuration.
