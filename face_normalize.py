"""Face normalisation utilities — rotation estimation, landmark alignment, and image warping.

Provides a unified pipeline to:
  1. Estimate head-pose rotation matrix R from MediaPipe 478-landmark anchors
  2. Decompose R into Euler angles (pitch, yaw, roll)
  3. Apply R⁻¹ to 3D landmarks → canonical front-facing position
  4. Apply the corresponding 2D affine or perspective warp to a video frame

Anchor landmarks (MediaPipe FaceMesh silhouette):
    10  — forehead (top of silhouette)
    152 — chin     (bottom of silhouette)
    1   — nose tip

Canonical orientation:
    Y-axis  = forehead → chin  (up = −Y)
    X-axis  = nose direction ⊥ Y  (depth/forward)
    Z-axis  = Y × X  (left–right)

Usage:
    from face_normalize import (
        estimate_rotation_matrix,
        rotation_to_euler,
        normalize_landmarks_3d,
        normalize_frame_affine,
        normalize_frame_perspective,
    )
"""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

# ── Anatomical anchor indices (MediaPipe FaceMesh 478) ───────────────────────
TOP_IDX = 10    # forehead
BOT_IDX = 152   # chin
NOSE_IDX = 1    # nose tip

# Additional stable anchors for the 2D warp (eye corners)
LEFT_EYE_OUTER = 263   # left-eye outer corner
RIGHT_EYE_OUTER = 33   # right-eye outer corner
LEFT_EYE_INNER = 362    # left-eye inner corner
RIGHT_EYE_INNER = 133   # right-eye inner corner

# 6-point set for a more robust affine (least-squares over-determined)
ANCHOR_6 = [TOP_IDX, BOT_IDX, NOSE_IDX,
            LEFT_EYE_OUTER, RIGHT_EYE_OUTER, 61]  # 61 = right mouth corner


# ═════════════════════════════════════════════════════════════════════════════
# §1  Rotation Matrix Estimation (3-point orthonormal basis)
# ═════════════════════════════════════════════════════════════════════════════

def estimate_rotation_matrix(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the 3×3 head-pose rotation matrix from 478 face landmarks.

    Builds an orthonormal face-basis from three anatomical anchors:
        Y_face = normalised(forehead 10 − chin 152)    face vertical (up)
        X_face = Gram–Schmidt(nose 1 − midpoint, Y)    face forward/depth
        Z_face = Y × X                                 face left–right

    The canonical (upright, front-facing) orientation in MediaPipe coords is:
        Y_canon = (0, -1,  0)   forehead up (y decreases)
        X_canon = (0,  0, -1)   nose toward camera (-z)
        Z_canon = (1,  0,  0)   left-right (+x)

    R_canonical is the face-basis matrix when the head is perfectly upright.
    R_current is the face-basis for the actual pose.
    The *deviation* matrix is:   R_dev = R_current @ R_canonical.T
    so that R_dev is identity when the head is in canonical pose and
    encodes roll/pitch/yaw deviations otherwise.

    Parameters
    ----------
    landmarks : (478, 3) float64
        Raw landmark coordinates (x, y, z).

    Returns
    -------
    R_dev : (3, 3) float64
        Deviation rotation matrix.  Identity = perfectly upright.
        To normalise, apply R_dev.T to the centred landmarks.
    center : (3,) float64
        Midpoint between forehead and chin (translation component).
    """
    center = (landmarks[TOP_IDX] + landmarks[BOT_IDX]) / 2.0
    pts = landmarks - center

    # Y-axis: chin → forehead (up)
    y_ax = pts[TOP_IDX] - pts[BOT_IDX]
    y_len = np.linalg.norm(y_ax)
    if y_len < 1e-12:
        return np.eye(3), center
    y_ax /= y_len

    # X-axis: nose tip direction, orthogonalised via Gram–Schmidt
    nose_dir = pts[NOSE_IDX]                 # relative to center
    nose_dir = nose_dir - np.dot(nose_dir, y_ax) * y_ax
    x_len = np.linalg.norm(nose_dir)
    if x_len < 1e-12:
        return np.eye(3), center
    x_ax = nose_dir / x_len

    # Z-axis: right-hand rule
    z_ax = np.cross(y_ax, x_ax)

    # R_current: rows are the face basis vectors in world coordinates
    R_current = np.vstack([x_ax, y_ax, z_ax])  # (3, 3)

    # Canonical basis (upright, front-facing in MediaPipe coordinate system)
    #   X_canon = (0,  0, -1)  nose toward camera
    #   Y_canon = (0, -1,  0)  forehead up
    #   Z_canon = (1,  0,  0)  left-right
    R_canonical = np.array([
        [ 0.0,  0.0, -1.0],   # X_canon
        [ 0.0, -1.0,  0.0],   # Y_canon
        [ 1.0,  0.0,  0.0],   # Z_canon
    ])

    # Deviation: how much the head has rotated away from canonical
    R_dev = R_current @ R_canonical.T
    return R_dev, center


# ═════════════════════════════════════════════════════════════════════════════
# §2  Euler Angle Decomposition
# ═════════════════════════════════════════════════════════════════════════════

def rotation_to_euler(R: np.ndarray, degrees: bool = True) -> tuple[float, float, float]:
    """Decompose a 3×3 rotation matrix into intrinsic Euler angles (ZYX convention).

    Returns (pitch, yaw, roll) where:
        pitch = rotation about X-axis  (nodding)
        yaw   = rotation about Y-axis  (turning left/right)
        roll  = rotation about Z-axis  (tilting head sideways)

    Parameters
    ----------
    R : (3, 3) rotation matrix
    degrees : bool
        If True, return angles in degrees; otherwise radians.

    Returns
    -------
    (pitch, yaw, roll) : tuple of float
    """
    # Clamp to avoid numerical issues with arcsin
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    yaw = math.asin(sy)

    if abs(sy) < 0.99999:
        pitch = math.atan2(R[2, 1], R[2, 2])
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        pitch = math.atan2(-R[1, 2], R[1, 1])
        roll = 0.0

    if degrees:
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        roll = math.degrees(roll)

    return pitch, yaw, roll


def euler_to_rotation_matrix(pitch: float, yaw: float, roll: float,
                             degrees: bool = True) -> np.ndarray:
    """Build a 3×3 rotation matrix from intrinsic ZYX Euler angles.

    This is the inverse of ``rotation_to_euler``: given (pitch, yaw, roll)
    it reconstructs the rotation matrix  R = Rz(roll) @ Ry(yaw) @ Rx(pitch).

    To *correct* a head pose, pass the negated angles::

        R_corr = euler_to_rotation_matrix(-pitch, -yaw, -roll)

    then apply  ``(R_corr @ centred_pts.T).T`` to the centred landmarks.

    Parameters
    ----------
    pitch : float   – rotation about X-axis (nodding)
    yaw   : float   – rotation about Y-axis (turning)
    roll  : float   – rotation about Z-axis (tilting)
    degrees : bool  – if True the angles are in degrees, else radians.

    Returns
    -------
    R : (3, 3) float64
    """
    if degrees:
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        roll = math.radians(roll)

    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cz, sz = math.cos(roll),  math.sin(roll)

    # Rx (pitch)
    Rx = np.array([[ 1,   0,   0],
                   [ 0,  cx, -sx],
                   [ 0,  sx,  cx]])
    # Ry (yaw)
    Ry = np.array([[ cy,  0,  sy],
                   [  0,  1,   0],
                   [-sy,  0,  cy]])
    # Rz (roll)
    Rz = np.array([[ cz, -sz,  0],
                   [ sz,  cz,  0],
                   [  0,   0,  1]])

    return Rz @ Ry @ Rx


def euler_correction_matrix(pitch: float, yaw: float, roll: float,
                            degrees: bool = True) -> np.ndarray:
    """Build the *inverse* rotation that cancels (pitch, yaw, roll).

    Since  R = Rz(roll) @ Ry(yaw) @ Rx(pitch),
    its inverse is  R⁻¹ = Rx(-pitch) @ Ry(-yaw) @ Rz(-roll)
    which equals negating each angle and composing in reverse order.

    This is the "multiply each angle by −1" correction matrix.

    Parameters
    ----------
    pitch, yaw, roll : float – the *measured* head-pose Euler angles
    degrees : bool

    Returns
    -------
    R_corr : (3, 3) float64 – apply ``(R_corr @ centred_pts.T).T``
    """
    if degrees:
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        roll = math.radians(roll)

    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cz, sz = math.cos(roll),  math.sin(roll)

    # Individual negated rotations
    Rx_neg = np.array([[ 1,   0,    0],
                       [ 0,  cx,   sx],
                       [ 0, -sx,   cx]])  # Rx(-pitch)
    Ry_neg = np.array([[ cy,  0,  -sy],
                       [  0,  1,    0],
                       [ sy,  0,   cy]])  # Ry(-yaw)
    Rz_neg = np.array([[ cz,  sz,  0],
                       [-sz,  cz,  0],
                       [  0,   0,  1]])  # Rz(-roll)

    # Reverse order: Rx(-p) @ Ry(-y) @ Rz(-r)
    return Rx_neg @ Ry_neg @ Rz_neg


def measure_angles_from_planes(landmarks: np.ndarray,
                               *,
                               degrees: bool = True) -> dict:
    """Measure pitch, yaw, and roll using XZ/YZ reference planes.

    Steps:
      1) Compute nose-tip distance to origin and use it as scale factor.
      2) Translate all points so nose tip (idx=1) is at (0,0,0).
      3) Scale all points by that distance.
      4) Measure:
         - pitch from forehead->chin projected to YZ plane (x=0)
         - yaw   from nose direction projected to XZ plane (y=0)
         - roll  from eye-line tilt in XY plane (z ignored)
    """
    if landmarks.shape[1] != 3:
        raise ValueError("landmarks must have shape (N, 3)")

    nose_world = landmarks[NOSE_IDX].astype(np.float64)
    nose_dist = float(np.linalg.norm(nose_world))
    if nose_dist < 1e-12:
        nose_dist = 1.0

    pts_nose = landmarks.astype(np.float64) - nose_world
    pts = pts_nose / nose_dist

    forehead = pts[TOP_IDX]
    chin = pts[BOT_IDX]
    right_eye = pts[RIGHT_EYE_OUTER]
    left_eye = pts[LEFT_EYE_OUTER]
    nose_tip = pts[NOSE_IDX]

    # Face vertical (forehead -> chin)
    v_fc = chin - forehead
    v_fc_norm = np.linalg.norm(v_fc)
    if v_fc_norm < 1e-12:
        v_fc = np.array([0.0, 1.0, 0.0])
    else:
        v_fc = v_fc / v_fc_norm

    # Face depth (mid forehead/chin -> nose)
    mid_fc = (forehead + chin) / 2.0
    v_nose = nose_tip - mid_fc
    v_nose_norm = np.linalg.norm(v_nose)
    if v_nose_norm < 1e-12:
        v_nose = np.array([0.0, 0.0, -1.0])
    else:
        v_nose = v_nose / v_nose_norm

    # Eye line (right -> left)
    v_eye = left_eye - right_eye
    v_eye_norm = np.linalg.norm(v_eye)
    if v_eye_norm < 1e-12:
        v_eye = np.array([1.0, 0.0, 0.0])
    else:
        v_eye = v_eye / v_eye_norm

    # Pitch: use YZ-plane projection of forehead->chin vector
    # canonical: chin lower than forehead => +Y direction for v_fc, pitch ~ 0
    vy, vz = v_fc[1], v_fc[2]
    pitch = math.atan2(vz, vy)

    # Yaw: use XZ-plane projection of nose direction
    nx, nz = v_nose[0], v_nose[2]
    yaw = math.atan2(nx, -nz)

    # Roll: eye-line tilt in XY plane
    ex, ey = v_eye[0], v_eye[1]
    roll = math.atan2(ey, ex)

    if degrees:
        pitch_out = math.degrees(pitch)
        yaw_out = math.degrees(yaw)
        roll_out = math.degrees(roll)
    else:
        pitch_out = pitch
        yaw_out = yaw
        roll_out = roll

    return {
        'pitch': pitch_out,
        'yaw': yaw_out,
        'roll': roll_out,
        'pitch_rad': pitch,
        'yaw_rad': yaw,
        'roll_rad': roll,
        'nose_dist': nose_dist,
        'nose_origin': nose_world,
        'landmarks_nose_scaled': pts,
    }


def correct_landmarks_from_planes(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], float]:
    """Correct pose via 3D matrices from plane-measured pitch/yaw/roll.

    Returns
    -------
    corrected : (N, 3)
    R_corr    : (3, 3)
    angles    : (pitch, yaw, roll) in degrees
    nose_dist : float  (distance of nose to world origin before centering)
    """
    info = measure_angles_from_planes(landmarks, degrees=True)
    pitch, yaw, roll = info['pitch'], info['yaw'], info['roll']

    # Build inverse correction from negative angles
    R_corr = euler_correction_matrix(pitch, yaw, roll, degrees=True)

    pts = info['landmarks_nose_scaled']
    corrected_scaled = (R_corr @ pts.T).T
    corrected = corrected_scaled * info['nose_dist'] + info['nose_origin']

    return corrected, R_corr, (pitch, yaw, roll), info['nose_dist']


# ═════════════════════════════════════════════════════════════════════════════
# §3  3D Landmark Normalisation
# ═════════════════════════════════════════════════════════════════════════════

def normalize_landmarks_3d(landmarks: np.ndarray,
                           R: np.ndarray | None = None,
                           center: np.ndarray | None = None) -> np.ndarray:
    """Apply R_dev⁻¹ to bring 3D landmarks to canonical front-facing position.

    R_dev is the deviation rotation matrix from estimate_rotation_matrix().
    Applying R_dev.T (= R_dev⁻¹) removes the head rotation so that the
    forehead→chin axis aligns with (0, -1, 0) and the nose points at -Z.

    If R and center are not provided, they are estimated from the landmarks.

    Parameters
    ----------
    landmarks : (N, 3) float64
    R : (3, 3) deviation rotation matrix (identity = canonical pose)
    center : (3,) translation (midpoint of forehead and chin)

    Returns
    -------
    normalised : (N, 3) float64  — landmarks in canonical pose
    """
    if R is None or center is None:
        R, center = estimate_rotation_matrix(landmarks)

    pts = landmarks - center
    # Apply inverse rotation: R.T (since R is orthonormal)
    normalised = (R.T @ pts.T).T + center
    return normalised


# ═════════════════════════════════════════════════════════════════════════════
# §4  2D Affine Warp  (for video frames)
# ═════════════════════════════════════════════════════════════════════════════

def _landmarks_to_pixels(landmarks: np.ndarray, h: int, w: int,
                         indices: list[int]) -> np.ndarray:
    """Convert normalised [0,1] landmark coords to pixel coordinates.

    MediaPipe landmarks have x ∈ [0, 1] (width), y ∈ [0, 1] (height).
    Returns (N, 2) float32 array of (px_x, px_y).
    """
    pts = landmarks[indices]
    px = np.column_stack([pts[:, 0] * w, pts[:, 1] * h]).astype(np.float32)
    return px


def _canonical_anchor_positions(landmarks: np.ndarray,
                                R: np.ndarray,
                                center: np.ndarray,
                                h: int, w: int,
                                indices: list[int]) -> np.ndarray:
    """Compute where anchor landmarks *should* be after de-rotation.

    Steps:
    1. Take current pixel positions of anchor landmarks
    2. Convert to normalised 3D
    3. Apply R.T (inverse rotation) around the center
    4. Project back to pixel coords (x*w, y*h)
    """
    # Normalise the full landmark set
    norm_lm = normalize_landmarks_3d(landmarks, R, center)
    # Extract the target pixel positions
    pts = norm_lm[indices]
    px = np.column_stack([pts[:, 0] * w, pts[:, 1] * h]).astype(np.float32)
    return px


def compute_affine_matrix(landmarks: np.ndarray,
                          h: int, w: int,
                          R: np.ndarray | None = None,
                          center: np.ndarray | None = None) -> np.ndarray:
    """Compute the 2×3 affine transformation matrix that de-rotates the face.

    Uses 3 anchor points: forehead (10), chin (152), nose tip (1).
    Returns M suitable for cv2.warpAffine(img, M, (w, h)).
    """
    if R is None or center is None:
        R, center = estimate_rotation_matrix(landmarks)

    anchor_idx = [TOP_IDX, BOT_IDX, NOSE_IDX]
    src_pts = _landmarks_to_pixels(landmarks, h, w, anchor_idx)
    dst_pts = _canonical_anchor_positions(landmarks, R, center, h, w, anchor_idx)

    M = cv2.getAffineTransform(src_pts, dst_pts)
    return M


def compute_affine_matrix_robust(landmarks: np.ndarray,
                                 h: int, w: int,
                                 R: np.ndarray | None = None,
                                 center: np.ndarray | None = None) -> np.ndarray:
    """Compute a 2×3 affine matrix from 6+ anchor points (least-squares).

    More stable than the exact 3-point version for noisy detections.
    Uses cv2.estimateAffinePartial2D for a similarity (4-DOF) or
    cv2.estimateAffine2D for full 6-DOF affine.
    """
    if R is None or center is None:
        R, center = estimate_rotation_matrix(landmarks)

    src_pts = _landmarks_to_pixels(landmarks, h, w, ANCHOR_6)
    dst_pts = _canonical_anchor_positions(landmarks, R, center, h, w, ANCHOR_6)

    # estimateAffine2D returns (M, inliers) — full 6-DOF affine, RANSAC
    M, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
    return M


def normalize_frame_affine(image: np.ndarray,
                           landmarks: np.ndarray,
                           R: np.ndarray | None = None,
                           center: np.ndarray | None = None,
                           robust: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """De-rotate a video frame using a 2D affine warp.

    Parameters
    ----------
    image : (H, W, 3) uint8 BGR or RGB frame
    landmarks : (478, 3) float64  normalised landmarks [0, 1]
    R, center : optional pre-computed rotation + translation
    robust : use 6-point LMEDS fit instead of exact 3-point

    Returns
    -------
    warped : (H, W, 3) same dtype/colour-space as input
    M : (2, 3) float64  the affine matrix that was applied
    """
    h, w = image.shape[:2]
    if R is None or center is None:
        R, center = estimate_rotation_matrix(landmarks)

    if robust:
        M = compute_affine_matrix_robust(landmarks, h, w, R, center)
    else:
        M = compute_affine_matrix(landmarks, h, w, R, center)

    warped = cv2.warpAffine(image, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return warped, M


# ═════════════════════════════════════════════════════════════════════════════
# §5  2D Perspective Warp  (handles large yaw/pitch)
# ═════════════════════════════════════════════════════════════════════════════

def compute_perspective_matrix(landmarks: np.ndarray,
                               h: int, w: int,
                               R: np.ndarray | None = None,
                               center: np.ndarray | None = None) -> np.ndarray:
    """Compute a 3×3 perspective (homography) matrix that de-rotates the face.

    Uses 4 anchor points: forehead, chin, left-eye outer, right-eye outer.
    Returns H suitable for cv2.warpPerspective(img, H, (w, h)).
    """
    if R is None or center is None:
        R, center = estimate_rotation_matrix(landmarks)

    anchor_idx = [TOP_IDX, BOT_IDX, LEFT_EYE_OUTER, RIGHT_EYE_OUTER]
    src_pts = _landmarks_to_pixels(landmarks, h, w, anchor_idx)
    dst_pts = _canonical_anchor_positions(landmarks, R, center, h, w, anchor_idx)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return H


def compute_perspective_matrix_robust(landmarks: np.ndarray,
                                      h: int, w: int,
                                      R: np.ndarray | None = None,
                                      center: np.ndarray | None = None) -> np.ndarray:
    """Compute a 3×3 homography from 6+ anchors using RANSAC.

    More robust than exact 4-point for noisy landmark detections.
    """
    if R is None or center is None:
        R, center = estimate_rotation_matrix(landmarks)

    src_pts = _landmarks_to_pixels(landmarks, h, w, ANCHOR_6)
    dst_pts = _canonical_anchor_positions(landmarks, R, center, h, w, ANCHOR_6)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
    if H is None:
        H = cv2.getPerspectiveTransform(src_pts[:4], dst_pts[:4])
    return H


def normalize_frame_perspective(image: np.ndarray,
                                landmarks: np.ndarray,
                                R: np.ndarray | None = None,
                                center: np.ndarray | None = None,
                                robust: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """De-rotate a video frame using a 2D perspective (homography) warp.

    Handles large yaw and pitch better than affine, but may introduce
    slight trapezoid distortion.

    Parameters
    ----------
    image : (H, W, 3) uint8 frame
    landmarks : (478, 3) float64
    R, center : optional pre-computed rotation + translation
    robust : use 6-point RANSAC fit instead of exact 4-point

    Returns
    -------
    warped : (H, W, 3)
    H : (3, 3) float64  the perspective matrix
    """
    h, w = image.shape[:2]
    if R is None or center is None:
        R, center = estimate_rotation_matrix(landmarks)

    if robust:
        H = compute_perspective_matrix_robust(landmarks, h, w, R, center)
    else:
        H = compute_perspective_matrix(landmarks, h, w, R, center)

    warped = cv2.warpPerspective(image, H, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped, H


# ═════════════════════════════════════════════════════════════════════════════
# §6  Convenience — full pipeline for a single frame
# ═════════════════════════════════════════════════════════════════════════════

def normalize_frame(image: np.ndarray,
                    landmarks: np.ndarray,
                    warp: Literal['affine', 'perspective'] = 'affine',
                    robust: bool = True,
                    ) -> dict:
    """One-call normalisation: estimate pose, warp image, normalise landmarks.

    Parameters
    ----------
    image : (H, W, 3) uint8 frame
    landmarks : (478, 3) float64  normalised [0, 1]
    warp : 'affine' or 'perspective'
    robust : multi-point LMEDS/RANSAC vs exact few-point

    Returns
    -------
    dict with keys:
        'image'        : (H, W, 3) warped frame
        'landmarks_3d' : (478, 3) de-rotated landmarks
        'R'            : (3, 3) rotation matrix (canonical → current)
        'center'       : (3,) translation
        'euler_deg'    : (pitch, yaw, roll) in degrees
        'warp_matrix'  : (2, 3) or (3, 3) transformation matrix applied
    """
    R, center = estimate_rotation_matrix(landmarks)
    euler = rotation_to_euler(R, degrees=True)
    norm_lm = normalize_landmarks_3d(landmarks, R, center)

    if warp == 'perspective':
        warped, M = normalize_frame_perspective(image, landmarks, R, center, robust)
    else:
        warped, M = normalize_frame_affine(image, landmarks, R, center, robust)

    return {
        'image': warped,
        'landmarks_3d': norm_lm,
        'R': R,
        'center': center,
        'euler_deg': euler,
        'warp_matrix': M,
    }
