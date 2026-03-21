"""Batch-extract audio (WAV) and face landmarks (JSONL) from video files.

For each .mp4 in dataset/sample/:
  1. Extract audio to dataset/sample/<name>.wav  using ffmpeg
  2. Extract per-frame face landmarks to dataset/sample/<name>.jsonl  using MediaPipe
  3. Compute head-pose rotation matrix R and Euler angles per frame
  4. Optionally apply affine/perspective warp to de-rotate frames

Usage:
    python test_video.py                          # process all .mp4 in dataset/sample/
    python test_video.py dataset/sample/00001.mp4 # process a single file
"""
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from face_normalize import (
    estimate_rotation_matrix,
    rotation_to_euler,
)

MODEL_PATH = 'face_landmarker_v2_with_blendshapes.task'
SAMPLE_DIR = Path('dataset/sample')


def extract_audio(video_path: Path, wav_path: Path, ffmpeg_path: str) -> bool:
    """Extract audio from a video file to a mono 16 kHz WAV using ffmpeg."""
    cmd = [
        ffmpeg_path, '-y',
        '-i', str(video_path),
        '-vn',                     # no video
        '-ac', '1',                # mono
        '-ar', '16000',            # 16 kHz
        '-acodec', 'pcm_s16le',    # PCM 16-bit
        str(wav_path),
    ]
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f'  ⚠ ffmpeg failed for {video_path.name}: {e}')
        return False


def extract_landmarks(video_path: Path, jsonl_path: Path, detector) -> int:
    """Run MediaPipe FaceLandmarker on every frame and write JSONL.

    Each line: {"t": <seconds>, "frame": <int>, "landmarks": [[{x,y,z}, ...]],
                "R": [[...], ...], "euler_deg": [pitch, yaw, roll]}
    Returns the number of frames processed.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'  ⚠ Cannot open video: {video_path.name}')
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)

            try:
                result = detector.detect(mp_image)
            except Exception:
                result = None

            entry = {
                't': frame_idx / fps,
                'frame': frame_idx,
                'landmarks': [],
                'R': None,
                'euler_deg': None,
            }

            if result is not None:
                faces = getattr(result, 'face_landmarks', []) or []
                for face in faces:
                    if hasattr(face, 'landmark'):
                        lm_iter = face.landmark
                    else:
                        try:
                            lm_iter = list(face)
                        except Exception:
                            lm_iter = []

                    lm_list = []
                    for lm in lm_iter:
                        lm_list.append({
                            'x': getattr(lm, 'x', None),
                            'y': getattr(lm, 'y', None),
                            'z': getattr(lm, 'z', None),
                        })
                    entry['landmarks'].append(lm_list)

                # Compute rotation matrix and Euler angles for the first face
                if entry['landmarks'] and len(entry['landmarks'][0]) >= 478:
                    face_pts = entry['landmarks'][0]
                    pts_arr = np.array([[p['x'] or 0, p['y'] or 0, p['z'] or 0]
                                        for p in face_pts], dtype='float64')
                    R, _center = estimate_rotation_matrix(pts_arr)
                    pitch, yaw, roll = rotation_to_euler(R, degrees=True)
                    entry['R'] = R.tolist()
                    entry['euler_deg'] = [round(pitch, 2), round(yaw, 2), round(roll, 2)]

            f.write(json.dumps(entry) + '\n')
            frame_idx += 1

    cap.release()
    return frame_idx


def main():
    # Resolve ffmpeg
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        print('ERROR: ffmpeg not found on PATH.')
        return

    # Resolve model
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        print(f'ERROR: Model not found: {MODEL_PATH}')
        return

    # Create detector once and reuse for all videos
    base_options = python.BaseOptions(model_asset_path=str(model_file))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Collect video files
    if len(sys.argv) > 1:
        videos = [Path(p) for p in sys.argv[1:]]
    else:
        videos = sorted(SAMPLE_DIR.glob('*.mp4'))

    if not videos:
        print(f'No .mp4 files found in {SAMPLE_DIR}')
        return

    print(f'Processing {len(videos)} video(s)…\n')

    for vp in videos:
        stem = vp.stem                  # e.g. "00001"
        out_dir = vp.parent             # same folder as source
        wav_path = out_dir / f'{stem}.wav'
        jsonl_path = out_dir / f'{stem}.jsonl'

        print(f'── {vp.name} ──')

        # 1. Audio extraction
        t0 = time.time()
        ok = extract_audio(vp, wav_path, ffmpeg_path)
        if ok:
            print(f'  Audio  → {wav_path}  ({time.time()-t0:.1f}s)')
        else:
            print(f'  Audio extraction failed (no audio track?)')

        # 2. Landmark extraction
        t0 = time.time()
        n = extract_landmarks(vp, jsonl_path, detector)
        print(f'  Landmarks → {jsonl_path}  ({n} frames, {time.time()-t0:.1f}s)')

    print('\nDone.')


if __name__ == '__main__':
    main()
