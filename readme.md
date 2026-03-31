# Face Landmarker Streamer

This repository contains utilities and small apps for running MediaPipe FaceLandmarker
on static images and live camera streams. The workspace was refactored to centralize
drawing utilities and archive rarely-used helper scripts.

Overview
- `face_utils.py`: shared drawing helper (`draw_landmarks_on_image`) used by other scripts.
- `test_static.py`: run the FaceLandmarker on a single image and write an annotated image.
- `test_stream_cv.py`: OpenCV streamer + recorder with optional microphone capture and
  automatic ffmpeg muxing (press `r` to start/stop recording, `q` to quit).

Quick start

1. Activate your venv and ensure dependencies are installed (OpenCV, mediapipe, sounddevice).

    ```bash
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. Download or place the MediaPipe model `face_landmarker_v2_with_blendshapes.task` in the repo root.

3. Run the static annotator:

    ```bash
    python test_static.py path/to/image.jpg face_landmarker_v2_with_blendshapes.task
    ```

4. Run the OpenCV streamer (camera index, model path, optional audio device index):

    ```bash
    python test_stream_cv.py 0 face_landmarker_v2_with_blendshapes.task
    ```

Notes
- Archived scripts are intact under `archive/` for reference or restoration.
- The `face_utils.draw_landmarks_on_image` API expects an RGB numpy array and
  a FaceLandmarker detection result object.

Tests
- A small unit test suite (`tests/`) is included to validate the drawing helper.
  Run tests with:

    ```bash
    pytest -q
    ```

If you want me to commit these changes to git or add CI (GitHub Actions), tell me which branch to use.
# Stream face mesh (MediaPipe Tasks)

Quick notes to run the real-time face mesh streamer (`test_stream.py`):

Prerequisites
- Activate your virtual environment (the repo uses `.venv`):

```bash
source .venv/bin/activate
```

- Ensure Python packages in `requirements.txt` are installed in the venv.

Model
- Place `face_landmarker_v2_with_blendshapes.task` in the repository root, or pass its path as the second argument to the script.
- You can download the model from MediaPipe assets; a canonical URL is:
  `https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2_with_blendshapes.task`

Run
- Use the repo venv Python to run the streamer (camera 0):

```bash
python test_stream.py
```

- Or pass camera index and model path:

```bash
python test_stream.py 0 face_landmarker_v2_with_blendshapes.task
```
