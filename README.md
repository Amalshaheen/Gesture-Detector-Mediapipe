##Pending Task
- Backward Gesture
- Testing

## Telegram Emergency Alert Setup

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

## Telegram Troubleshooting (409 Conflict)

If you see `HTTP Error 409: Conflict`, Telegram is usually in webhook mode or another client is calling `getUpdates`.

1. Disable webhook for this bot:

```bash
curl -s "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/deleteWebhook?drop_pending_updates=true"
```

2. Open Telegram and send `/start` (or any message) to your bot.

3. Prefer setting `TELEGRAM_CHAT_ID` directly in `.env` so the app does not need `getUpdates` lookup.