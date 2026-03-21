"""Shared face utilities (drawing helpers) used by the static and streaming scripts.

Contains `draw_landmarks_on_image` which draws Mediapipe FaceLandmarker landmarks
and connections onto an RGB numpy image using OpenCV.
"""
try:
    from mediapipe.tasks.python import vision
except Exception:
    vision = None

import numpy as np
import cv2


def _get_connection_pairs(connections_obj):
    """Return an iterable of (start, end) integer index pairs from a connections object.
    The `vision.FaceLandmarksConnections` members may be sequences of pairs or objects
    with `start`/`end` attributes; handle both forms.
    """
    if connections_obj is None:
        return []
    try:
        return list(connections_obj)
    except Exception:
        return []


def draw_landmarks_on_image(rgb_image, detection_result, point_color=(0,255,0), line_color=(255,0,0)):
    """Draw landmarks and connections on `rgb_image` using OpenCV.

    - `rgb_image` is a HxWx3 NumPy array in RGB order.
    - `detection_result.face_landmarks` is iterated; each face_landmarks may expose
      a `.landmark` sequence or be directly iterable.
    """
    face_landmarks_list = getattr(detection_result, 'face_landmarks', []) or []
    annotated_image = np.copy(rgb_image)
    h, w = annotated_image.shape[:2]

    # Prepare connection lists. If MediaPipe's vision module isn't available
    # (for unit tests run in minimal environments), fall back to empty lists.
    if vision is None:
        tess = []
        contours = []
    else:
        tess = _get_connection_pairs(getattr(vision.FaceLandmarksConnections, 'FACE_LANDMARKS_TESSELATION', None))
        contours = _get_connection_pairs(getattr(vision.FaceLandmarksConnections, 'FACE_LANDMARKS_CONTOURS', None))

    for face_landmarks in face_landmarks_list:
        # Extract landmark iterable
        if hasattr(face_landmarks, 'landmark'):
            landmarks_iter = face_landmarks.landmark
        else:
            try:
                landmarks_iter = list(face_landmarks)
            except Exception:
                landmarks_iter = []

        pts = []
        for lm in landmarks_iter:
            # Landmark may have normalized x/y in [0,1]
            lx = getattr(lm, 'x', None)
            ly = getattr(lm, 'y', None)
            if lx is None or ly is None:
                continue
            px = int(lx * w)
            py = int(ly * h)
            pts.append((px, py))
            # Draw slightly larger points for visibility
            cv2.circle(annotated_image, (px, py), 2, point_color, -1)

        # Draw connection lines if points exist
        def draw_connections(conns):
            for conn in conns:
                try:
                    start, end = conn
                except Exception:
                    try:
                        start = getattr(conn, 'start')
                        end = getattr(conn, 'end')
                    except Exception:
                        continue
                if start < len(pts) and end < len(pts):
                    # Use a thicker line so mesh is visible on webcam
                    cv2.line(annotated_image, pts[start], pts[end], line_color, 2)

        draw_connections(tess)
        draw_connections(contours)

    return annotated_image
