# S25 AI Camera v0.2

Experimental camera for Galaxy S25. Photos are captured with CameraX and automatically enhanced on-device with a Real-ESRGAN x2 ONNX neural network.

## Build
Push to `main`. GitHub Actions downloads the model, builds the debug APK, and publishes `S25-AI-Camera-v0.2` under the workflow run's Artifacts section.

## Notes
- AI processing is local after installation; the APK contains the model.
- v0.2 processes a central high-detail region at large zoom to keep memory/processing time practical on a phone.
- Real-ESRGAN can reconstruct plausible detail; enhanced pixels should not be treated as forensic evidence.
