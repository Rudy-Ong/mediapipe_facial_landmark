"""OpenCV-based streamer + recorder
The video stream also displayed face landmarks in real-time
Controls:
- Press 'r' to start/stop recording
- Press 'q' or ESC to quit

Saves recordings to `output/stream/`.
"""
import time
import json
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from face_utils import draw_landmarks_on_image
from audio_filter import bandpass_filter_wav
from face_normalize import estimate_rotation_matrix, rotation_to_euler
import os
import shutil
import subprocess
import wave
import numpy as np


def main(camera_idx=0, model_path='face_landmarker_v2_with_blendshapes.task', audio_device=None):
    model_file = Path(model_path)
    if not model_file.exists():
        print(f'Model not found: {model_path}')
        return

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f'Unable to open camera {camera_idx}')
        return

    # Create detector
    try:
        base_options = python.BaseOptions(model_asset_path=str(model_file))
        options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=False, output_facial_transformation_matrixes=False, num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        print('Failed to create FaceLandmarker:', e)
        cap.release()
        return

    # Verify ffmpeg is available — required for audio recording
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        print('ERROR: ffmpeg not found on PATH. Audio recording requires ffmpeg.')
        cap.release()
        return

    is_recording = False
    writer = None
    record_start = None
    meta_file = None
    recording_timestamp = None
    out_path = None          # current video output path
    record_fps = 25        # set when recording starts Physical Frame Rate Macbook M2
    frame_count = 0          # frames written since recording started
    # Audio state (ffmpeg subprocess)
    audio_proc = None       # ffmpeg subprocess recording mic -> WAV
    wav_path = None          # path to the WAV being recorded
    recent_samples = np.zeros(1, dtype='float32')

    def _read_wav_tail(path, n_samples=4096):
        """Read the last `n_samples` samples from a WAV file for waveform display."""
        try:
            with wave.open(str(path), 'rb') as wf:
                nframes = wf.getnframes()
                sw = wf.getsampwidth()
                ch = wf.getnchannels()
                if nframes == 0:
                    return np.zeros(n_samples, dtype='float32')
                read_start = max(0, nframes - n_samples)
                wf.setpos(read_start)
                raw = wf.readframes(nframes - read_start)
                if sw == 2:
                    data = np.frombuffer(raw, dtype='<i2').astype('float32') / 32767.0
                elif sw == 4:
                    data = np.frombuffer(raw, dtype='<i4').astype('float32') / 2147483647.0
                else:
                    data = np.frombuffer(raw, dtype='uint8').astype('float32') / 128.0 - 1.0
                if ch > 1:
                    data = data[::ch]  # take first channel
                return data[-n_samples:] if data.size > n_samples else data
        except Exception:
            return np.zeros(n_samples, dtype='float32')

    def _build_ffmpeg_audio_cmd(wav_output, device_index=None):
        """Build an ffmpeg command list to record from the default mic to a WAV file."""
        cmd = [ffmpeg_path, '-y']
        # macOS: use avfoundation, audio-only (":device_index")
        dev = f':{int(device_index)}' if device_index is not None else ':0'
        cmd += ['-f', 'avfoundation', '-i', dev]
        cmd += ['-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le', str(wav_output)]
        return cmd

    out_dir = Path('output/stream')
    out_dir.mkdir(parents=True, exist_ok=True)

    window = 'Face Recorder (press r to start/stop, q to quit)'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print('Controls: r=start/stop recording, q=quit')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Frame read failed, exiting')
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detection_result = None
        try:
            # Construct mediapipe Image from numpy array (supported by this mediapipe version)
            mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
            detection_result = detector.detect(mp_image)
            annotated_rgb = draw_landmarks_on_image(rgb, detection_result)
        except Exception:
            annotated_rgb = rgb

        # Overlay a simple face count indicator so it's obvious whether detection ran
        try:
            face_count = len(getattr(detection_result, 'face_landmarks', []) or [])
        except Exception:
            face_count = 0

        # Read recent audio samples from WAV file for live waveform overlay
        try:
            wave_width = int(annotated_rgb.shape[1])
        except Exception:
            wave_width = 320

        if is_recording and wav_path is not None and wav_path.exists():
            recent_samples = _read_wav_tail(wav_path, n_samples=wave_width)

        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Overlay recording status, timer, and face count
        if is_recording and record_start is not None:
            elapsed = int(time.time() - record_start)
            h = elapsed // 3600
            m = (elapsed % 3600) // 60
            s = elapsed % 60
            timer_text = f'Recording {h:02d}:{m:02d}:{s:02d}'
            cv2.putText(annotated_bgr, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            cv2.circle(annotated_bgr, (160,24), 6, (0,0,255), -1)
        else:
            cv2.putText(annotated_bgr, 'Press r to record', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        # Show face detection count
        try:
            cv2.putText(annotated_bgr, f'Faces: {face_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        except Exception:
            pass

        cv2.imshow(window, annotated_bgr)

        if is_recording and writer is not None:
            writer.write(annotated_bgr)
            # Also write landmark info to metadata file (JSON lines)
            if meta_file is not None:
                entry = {
                    't': frame_count / record_fps,
                    'frame': frame_count,
                    'landmarks': [],
                    'R': None,
                    'euler_deg': None,
                }
                frame_count += 1
                try:
                    faces = getattr(detection_result, 'face_landmarks', []) if 'detection_result' in locals() else []
                    for face in faces:
                        lm_list = []
                        if hasattr(face, 'landmark'):
                            landmarks_iter = face.landmark
                        else:
                            try:
                                landmarks_iter = list(face)
                            except Exception:
                                landmarks_iter = []
                        for lm in landmarks_iter:
                            lx = getattr(lm, 'x', None)
                            ly = getattr(lm, 'y', None)
                            lz = getattr(lm, 'z', None)
                            lm_list.append({'x': lx, 'y': ly, 'z': lz})
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
                except Exception:
                    pass
                try:
                    meta_file.write(json.dumps(entry) + '\n')
                except Exception:
                    pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            # toggle recording
            if not is_recording:
                timestamp = time.strftime('%Y%m%d-%H%M%S')
                out_path = out_dir / f'stream_{timestamp}.mp4'
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or annotated_bgr.shape[1])
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or annotated_bgr.shape[0])
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                record_fps = fps
                frame_count = 0

                # --- Start audio FIRST so ffmpeg has time to initialise ---
                wav_path = out_dir / f'stream_{timestamp}.wav'
                recent_samples = np.zeros(width, dtype='float32')
                try:
                    audio_cmd = _build_ffmpeg_audio_cmd(wav_path, device_index=audio_device)
                    audio_proc = subprocess.Popen(
                        audio_cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    # Wait until ffmpeg creates the WAV file (up to 2 s)
                    for _ in range(40):
                        if wav_path.exists() and wav_path.stat().st_size > 44:
                            break
                        time.sleep(0.05)
                    print('Audio recording started (ffmpeg pid=%d)' % audio_proc.pid)
                except Exception as e:
                    print('Failed to start ffmpeg audio recording:', e)
                    audio_proc = None

                # --- Now start video writer ---
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    print('Failed to open video writer')
                    writer = None
                    # Kill the audio process we just started
                    if audio_proc is not None:
                        audio_proc.terminate()
                        audio_proc = None
                else:
                    is_recording = True
                    record_start = time.time()
                    recording_timestamp = timestamp
                    # Open metadata JSONL alongside the video
                    meta_path = out_dir / f'stream_{timestamp}.jsonl'
                    try:
                        meta_file = open(meta_path, 'w', encoding='utf8')
                    except Exception as e:
                        print('Failed to open metadata file:', e)
                        meta_file = None
                    print('Recording started:', out_path)
            else:
                is_recording = False
                if writer is not None:
                    writer.release()
                    print('Recording stopped')
                    writer = None
                    record_start = None
                    if meta_file is not None:
                        try:
                            meta_file.close()
                        except Exception:
                            pass
                        meta_file = None
                    # Stop ffmpeg audio recording and mux into MP4
                    if audio_proc is not None:
                        try:
                            # Send 'q' to ffmpeg stdin for graceful stop
                            audio_proc.stdin.write(b'q')
                            audio_proc.stdin.flush()
                            audio_proc.wait(timeout=5)
                        except Exception:
                            audio_proc.terminate()
                            try:
                                audio_proc.wait(timeout=3)
                            except Exception:
                                audio_proc.kill()
                        audio_proc = None
                        print('Audio recording stopped')

                    # Apply bandpass filter to remove noise before muxing
                    if wav_path is not None and wav_path.exists():
                        try:
                            bandpass_filter_wav(str(wav_path), low_hz=80, high_hz=15000, order=4)
                            print('Applied bandpass filter (80 Hz – 15 kHz) to', wav_path)
                        except Exception as e:
                            print('Bandpass filter failed (muxing unfiltered):', e)

                    # Mux WAV into MP4 (overwrite original)
                    if wav_path is not None and wav_path.exists() and out_path is not None:
                        tmp_out = out_dir / f'stream_{recording_timestamp}_tmp.mp4'
                        mux_cmd = [
                            ffmpeg_path, '-y',
                            '-i', str(out_path),
                            '-i', str(wav_path),
                            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                            str(tmp_out),
                        ]
                        try:
                            subprocess.run(mux_cmd, check=True,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
                            os.replace(str(tmp_out), str(out_path))
                            print('Muxed audio into video, overwritten', out_path)
                        except Exception as e:
                            print('ffmpeg muxing failed:', e)
                    wav_path = None
                    recent_samples = np.zeros(1, dtype='float32')

    # Cleanup
    if audio_proc is not None:
        try:
            audio_proc.terminate()
            audio_proc.wait(timeout=3)
        except Exception:
            audio_proc.kill()
    if writer is not None:
        writer.release()
    if meta_file is not None:
        try:
            meta_file.close()
        except Exception:
            pass
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model = sys.argv[2] if len(sys.argv) > 2 else 'face_landmarker_v2_with_blendshapes.task'
    main(camera_idx=cam, model_path=model)

