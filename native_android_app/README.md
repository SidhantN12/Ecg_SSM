# Native Android ECG Client

This Android app replaces the MQTT client path with a simple HTTP poll:

- Request: `GET http://<raspberry-pi-ip>:8000/latest`
- Action: tap **Fetch latest**
- Result: display diagnosis label and confidence

## What is included

- Editable Raspberry Pi base URL field
- One-button fetch flow
- Diagnosis and confidence display
- Basic error handling for bad URLs, network failures, and unexpected JSON

## Expected API response

The app is happiest with:

```json
{
  "label": "N (Normal)",
  "confidence": 0.97
}
```

It also accepts common alternatives like `diagnosis`, `prediction`, `probability`, `score`, or a nested `result` object.

## Open in Android Studio

1. Open the `native_android_app/` folder as a project.
2. Let Gradle sync.
3. Run the `app` configuration on an Android device or emulator.

## Notes

- The app uses plain HTTP because your Raspberry Pi endpoint is local LAN infrastructure.
- Android 9+ blocks cleartext by default, so the manifest explicitly enables cleartext traffic for this app.
