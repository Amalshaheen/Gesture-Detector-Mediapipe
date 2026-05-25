# Gesture Detector

Gesture Detector is a real time hand gesture recognition system that controls a wheelchair prototype. It uses MediaPipe hand landmarks, scores each gesture template, stabilizes noisy predictions, and sends commands to an ESP32 over Bluetooth. An emergency gesture triggers Telegram alerts.

## Project website

Open `index.html` in a browser to view a simple project overview with photos, the system diagram, and the gesture map.

## Key features

- Real time hand tracking with MediaPipe HandLandmarker
- 7 gestures mapped to wheelchair commands
- Command stabilization to avoid flicker
- Bluetooth serial control of ESP32 firmware
- Emergency gesture sends Telegram alerts
- Safety STOP on tracking loss or camera failure

## Gesture map

| Gesture | Hand pose | Command |
| --- | --- | --- |
| STOP | Open palm down | S |
| BACKWARD | Open palm up | B |
| FORWARD | Index only, palm down | F |
| LEFT | Thumb only, palm down | L |
| RIGHT | Little finger only, palm down | R |
| HORN | Index and middle down | H |
| EMERGENCY | Index and middle up | E |

Gesture reference photos are in `statics/gestures/`.

## Quick start

1. Install dependencies:

```bash
python -m pip install mediapipe opencv-python pyserial
```

2. Pair the ESP32 over Bluetooth (the default port is `/dev/rfcomm0`).

3. Run the app:

```bash
python main.py
```

If Bluetooth is unavailable, the app runs in demo mode and logs commands to the console.

## Telegram emergency alert setup

1. Create a `.env` file in the project root (or copy from `.env.example`):

```bash
cp .env.example .env
```

2. Add your Telegram values in `.env`:

```bash
TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID="YOUR_CHAT_ID"
STARTUP_ALERT_ENABLED="1"
```

Optional: disable startup online notification:

```bash
STARTUP_ALERT_ENABLED="0"
```

3. If `TELEGRAM_CHAT_ID` is not set, the app auto-detects it from `getUpdates`.
To enable this, send at least one message to your bot from Telegram first.

4. Run the app:

```bash
python main.py
```

When the `EMERGENCY` gesture is detected, the system sends a Telegram alert.
When the gesture clears, it sends a recovery message.

## Telegram troubleshooting (409 conflict)

If you see `HTTP Error 409: Conflict`, Telegram is usually in webhook mode or another client is calling `getUpdates`.

1. Disable webhook for this bot:

```bash
curl -s "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/deleteWebhook?drop_pending_updates=true"
```

2. Open Telegram and send `/start` (or any message) to your bot.

3. Prefer setting `TELEGRAM_CHAT_ID` directly in `.env` so the app does not need `getUpdates` lookup.

## Documentation

- `CODE_DOCUMENTATION.md` contains the full code walkthrough.
- `GESTURE_PIPELINE_EXPLAINED.md` details the gesture scoring and stabilization pipeline.

## Hardware firmware

ESP32 logic lives in `hardwarecode.ino`. It reads single character commands and drives the motors, buzzer, and safety logic.