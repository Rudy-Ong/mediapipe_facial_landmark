"""Visualise face landmarks from a sample JSONL file in 3D — with head-tilt normalisation.

Head tilt is corrected by aligning the forehead→chin axis (landmarks 10→152)
to the pure Y-axis using a rigid rotation.

Seven correction modes are available (pass --method on the CLI):
1. rodrigues   – Rodrigues' rotation: align 10→152 to Y-axis
2. procrustes  – Procrustes alignment: rigidly fit every frame to frame-0
3. basis       – Three-point orthonormal basis (forehead 10, chin 152, nose 1)
4. affine      – Full rotation matrix + 2D affine warp  (shows Euler angles)
5. perspective – Full rotation matrix + 2D perspective warp  (shows Euler angles)
6. euler       – Negate pitch/yaw/roll → build R_correction → apply  (most intuitive)
7. planes      – Nose-origin + XZ/YZ planes + 3D matrix correction
8. frontalize  – Show Original, Affine-frontalized, Rigid-canonical

Usage:
    python sample_vis_no_tilt.py                          # interactive picker, planes
    python sample_vis_no_tilt.py 00001.jsonl              # specific file
    python sample_vis_no_tilt.py 00001.jsonl --method procrustes
    python sample_vis_no_tilt.py 00001.jsonl --method affine
"""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from face_normalize import (
    estimate_rotation_matrix,
    rotation_to_euler,
    euler_to_rotation_matrix,
    euler_correction_matrix,
    measure_angles_from_planes,
    correct_landmarks_from_planes,
    compute_affine_matrix_robust,
    normalize_landmarks_3d,
)

# ── Mesh annotation (MediaPipe FaceMesh 478-landmark indices) ────────────────
Mesh_annotation = {
    "silhouette": [
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109],

    "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],

    "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173],
    "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    # "rightEyeUpper1": [247, 30, 29, 27, 28, 56, 190],
    # "rightEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],
    # "rightEyeUpper2": [113, 225, 224, 223, 222, 221, 189],
    # "rightEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],
    # "rightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],

    # "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
    # "rightEyebrowLower": [35, 124, 46, 53, 52, 65],

    # "rightEyeIris": [473, 474, 475, 476, 477],

    "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
    "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
    # "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
    # "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
    # "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
    # "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
    # "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],

    # "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
    # "leftEyebrowLower": [265, 353, 276, 283, 282, 295],
    # "leftEyeIris": [468, 469, 470, 471, 472],

    # "midwayBetweenEyes": [168],

    "noseTip": [1],
    "noseBottom": [2],
    "noseRightCorner": [98],
    "noseLeftCorner": [327],

    # "rightCheek": [205],
    # "leftCheek": [425],
}

PARTS = {
    "silhouette":     ("white", 1.0),
    "lipsUpperOuter": ("red", 1.0),
    "lipsUpperInner": ("red", 1.0),
    "lipsLowerOuter": ("red", 1.0),
    "lipsLowerInner": ("red", 1.0),
    "rightEyeUpper0": ("cyan", 1.0),
    "rightEyeLower0": ("cyan", 1.0),
    # "rightEyeUpper1": ("cyan", 1.0),
    # "rightEyeLower1": ("cyan", 1.0),
    # "rightEyeUpper2": ("cyan", 1.0),
    # "rightEyeLower2": ("cyan", 1.0),
    # "rightEyeLower3": ("cyan", 1.0),
    # "rightEyebrowUpper": ("yellow", 1.0),
    # "rightEyebrowLower": ("yellow", 1.0),
    # "rightEyeIris": ("blue", 1.0),
    "leftEyeUpper0": ("green", 1.0),
    "leftEyeLower0": ("green", 1.0),
    # "leftEyeUpper1": ("green", 1.0),
    # "leftEyeLower1": ("green", 1.0),
    # "leftEyeUpper2": ("green", 1.0),
    # "leftEyeLower2": ("green", 1.0),
    # "leftEyeLower3": ("green", 1.0),
    # "leftEyebrowUpper": ("orange", 1.0),
    # "leftEyebrowLower": ("orange", 1.0),
    # "leftEyeIris": ("purple", 1.0),
    # "midwayBetweenEyes": ("white", 1.0),
    "noseTip": ("magenta", 1.0),
    "noseBottom": ("magenta", 1.0),
    "noseRightCorner": ("magenta", 1.0),
    "noseLeftCorner": ("magenta", 1.0),
    # "rightCheek": ("gray", 1.0),
    # "leftCheek": ("gray", 1.0),
}

TOP_IDX = 10    # forehead (top of silhouette)
BOT_IDX = 152   # chin     (bottom of silhouette)
NOSE_IDX = 1    # nose tip

# ═════════════════════════════════════════════════════════════════════════════
# Method 1 — Rodrigues' rotation  (align forehead→chin axis to Y-axis)
# ═════════════════════════════════════════════════════════════════════════════
def normalize_tilt_rodrigues(landmarks: np.ndarray) -> np.ndarray:
    """Rotate all 478 landmarks so that 10 (forehead) and 152 (chin) are
    vertically aligned along the Y-axis (same X and Z).

    Uses Rodrigues' rotation formula — a single rigid rotation that
    preserves all inter-landmark distances (lip shape stays accurate).
    """
    mid = (landmarks[TOP_IDX] + landmarks[BOT_IDX]) / 2.0
    pts = landmarks - mid

    # Current head axis: chin → forehead
    v = pts[TOP_IDX] - pts[BOT_IDX]
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return landmarks.copy()
    v = v / v_norm

    # Target axis: pure −Y  (MediaPipe Y increases downward, forehead < chin)
    target = np.array([0.0, -1.0, 0.0])

    # Rotation axis & angle via cross product
    axis = np.cross(v, target)
    sin_a = np.linalg.norm(axis)
    cos_a = np.dot(v, target)

    if sin_a < 1e-8:
        return landmarks.copy()   # already aligned

    axis = axis / sin_a           # unit rotation axis

    # Skew-symmetric matrix K for Rodrigues
    K = np.array([[       0, -axis[2],  axis[1]],
                  [ axis[2],        0, -axis[0]],
                  [-axis[1],  axis[0],        0]])

    R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

    return (R @ pts.T).T + mid


# ═════════════════════════════════════════════════════════════════════════════
# Method 2 — Procrustes alignment  (rigidly align each frame to frame 0)
# ═════════════════════════════════════════════════════════════════════════════
def _procrustes_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Optimal rotation (no scaling) from source → target via SVD."""
    H = source.T @ target
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, np.sign(d)])
    return Vt.T @ S @ U.T


def normalize_tilt_procrustes(landmarks: np.ndarray,
                              reference: np.ndarray) -> np.ndarray:
    """Rigidly rotate `landmarks` to best-fit `reference` (usually frame 0)."""
    src_c = landmarks.mean(axis=0)
    ref_c = reference.mean(axis=0)
    src_centred = landmarks - src_c
    ref_centred = reference - ref_c
    R = _procrustes_rotation(src_centred, ref_centred)
    return (R @ src_centred.T).T + ref_c


# ═════════════════════════════════════════════════════════════════════════════
# Method 3 — Three-point orthonormal basis  (forehead, chin, nose tip)
# ═════════════════════════════════════════════════════════════════════════════
def normalize_tilt_basis(landmarks: np.ndarray) -> np.ndarray:
    """Project landmarks into a face-centric coordinate system built from
    three anatomical anchors:
        Y-axis = normalised (forehead 10 − chin 152)       up
        X-axis = component of (nose 1 − midpoint) ⊥ Y      forward/depth
        Z-axis = Y × X                                     left–right

    This removes roll, pitch, AND yaw — the face always appears front-on.
    """
    mid = (landmarks[TOP_IDX] + landmarks[BOT_IDX]) / 2.0
    pts = landmarks - mid

    # Y-axis: chin → forehead
    y_ax = pts[TOP_IDX] - pts[BOT_IDX]
    y_len = np.linalg.norm(y_ax)
    if y_len < 1e-12:
        return landmarks.copy()
    y_ax = y_ax / y_len

    # X-axis: nose direction, made orthogonal to Y via Gram–Schmidt
    nose_dir = pts[NOSE_IDX] - (pts[TOP_IDX] + pts[BOT_IDX]) / 2.0
    nose_dir = nose_dir - np.dot(nose_dir, y_ax) * y_ax
    x_len = np.linalg.norm(nose_dir)
    if x_len < 1e-12:
        return landmarks.copy()
    x_ax = nose_dir / x_len

    # Z-axis: right-hand rule
    z_ax = np.cross(y_ax, x_ax)

    R = np.vstack([x_ax, y_ax, z_ax])   # (3, 3) — rows are new basis
    return (R @ pts.T).T + mid


# ═════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═════════════════════════════════════════════════════════════════════════════
def load_jsonl(path: str):
    times, landmarks = [], []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            t = entry.get('t')
            lm_faces = entry.get('landmarks', [])
            if t is None or len(lm_faces) == 0:
                continue
            face = lm_faces[0]
            pts = np.array([[p.get('x', 0), p.get('y', 0), p.get('z', 0)]
                            for p in face], dtype='float64')
            times.append(float(t))
            landmarks.append(pts)
    return times, landmarks


def pick_jsonl(args_rest):
    script_dir = Path(__file__).resolve().parent
    if args_rest:
        candidate = Path(args_rest[0])
        if not candidate.is_absolute():
            candidate = script_dir / candidate
        candidate = candidate.resolve()
        if candidate.is_file():
            return str(candidate)
        print(f'File not found: {candidate}')
        raise SystemExit(1)

    files = sorted(script_dir.glob('*.jsonl'))
    if not files:
        print('No .jsonl files found in this directory.')
        raise SystemExit(1)
    if len(files) == 1:
        return str(files[0])

    print('Available JSONL files:')
    for i, fp in enumerate(files, 1):
        print(f'  [{i}] {fp.name}')
    print()
    while True:
        try:
            choice = input(f'Select a file (1-{len(files)}): ').strip()
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return str(files[idx])
        except (ValueError, EOFError):
            pass
        print('Invalid choice, try again.')


# ═════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═════════════════════════════════════════════════════════════════════════════

def _style_3d_axis(ax):
    """Apply dark theme to a 3D axis."""
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color('white')
        axis.set_pane_color((0.12, 0.12, 0.12, 1.0))
        for tick in axis.get_ticklabels():
            tick.set_color('gray')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')


def _compute_axis_limits(frames_list, pad=0.05):
    """Compute shared axis limits from one or more frame arrays."""
    used_indices = sorted({idx for name in PARTS for idx in Mesh_annotation[name]})
    all_pts = np.vstack(frames_list)
    mins = all_pts[used_indices].min(axis=0)
    maxs = all_pts[used_indices].max(axis=0)
    return ((mins[0] - pad, maxs[0] + pad),
            (maxs[1] + pad, mins[1] - pad),
            (mins[2] - pad, maxs[2] + pad))


def _frontalize_landmarks_affine(landmarks: np.ndarray) -> np.ndarray:
    """Frontalize one frame using affine warp on (x,y) + de-rotated z."""
    R, center = estimate_rotation_matrix(landmarks)
    M = compute_affine_matrix_robust(landmarks, h=1, w=1, R=R, center=center)

    ones = np.ones((landmarks.shape[0], 1), dtype=np.float64)
    src = np.hstack([landmarks[:, :2], ones])
    dst_xy = (M @ src.T).T

    norm3d = normalize_landmarks_3d(landmarks, R, center)
    out = landmarks.copy()
    out[:, :2] = dst_xy
    out[:, 2] = norm3d[:, 2]
    return out


def _frontalize_landmarks_rigid(landmarks: np.ndarray) -> np.ndarray:
    """Frontalize one frame using pure 3D rigid rotation to canonical pose."""
    R, center = estimate_rotation_matrix(landmarks)
    return normalize_landmarks_3d(landmarks, R, center)


def _make_draw_frame(ax, frames, x_lim, y_lim, z_lim, plane_infos=None,
                     axis_mats=None):
    """Return a (draw_frame, artists) pair for a single 3D subplot."""
    artists = []

    def draw_frame(fi):
        for a in artists:
            a.remove()
        artists.clear()
        fd = frames[fi]
        for name, (colour, alpha) in PARTS.items():
            idx = Mesh_annotation[name]
            xs, ys, zs = fd[idx, 0], fd[idx, 1], fd[idx, 2]
            artists.append(ax.scatter(xs, ys, zs, c=colour, alpha=alpha,
                                      s=8, edgecolors='none'))
            line, = ax.plot(xs, ys, zs, color=colour,
                            alpha=alpha * 0.7, linewidth=1)
            artists.append(line)
        # Highlight the tilt axis (10 ◆ → 152 ◆)
        for pt in [TOP_IDX, BOT_IDX]:
            artists.append(ax.scatter(*fd[pt], c='lime', s=40,
                                      marker='D', edgecolors='white',
                                      linewidths=0.5))
        ln, = ax.plot([fd[TOP_IDX, 0], fd[BOT_IDX, 0]],
                      [fd[TOP_IDX, 1], fd[BOT_IDX, 1]],
                      [fd[TOP_IDX, 2], fd[BOT_IDX, 2]],
                      color='lime', linewidth=2, linestyle='--', alpha=0.8)
        artists.append(ln)

        # Optional metadata hook for plane-based mode (kept for compatibility).
        if plane_infos is not None:
            nose = fd[NOSE_IDX]
            x0, y0, z0 = nose[0], nose[1], nose[2]

            # Nose origin marker
            artists.append(ax.scatter(x0, y0, z0, c='white', s=30,
                                      marker='o', edgecolors='black',
                                      linewidths=0.6))

        # Optional: draw per-frame pose axes (pitch/yaw/roll axes) at nose.
        if axis_mats is not None:
            nose = fd[NOSE_IDX]
            R_axis = axis_mats[fi]
            axis_len = 0.07

            x_axis = R_axis @ np.array([axis_len, 0.0, 0.0])
            y_axis = R_axis @ np.array([0.0, axis_len, 0.0])
            z_axis = R_axis @ np.array([0.0, 0.0, axis_len])

            qx = ax.quiver(nose[0], nose[1], nose[2],
                           x_axis[0], x_axis[1], x_axis[2],
                           color='red', linewidth=2.0, arrow_length_ratio=0.18)
            qy = ax.quiver(nose[0], nose[1], nose[2],
                           y_axis[0], y_axis[1], y_axis[2],
                           color='lime', linewidth=2.0, arrow_length_ratio=0.18)
            qz = ax.quiver(nose[0], nose[1], nose[2],
                           z_axis[0], z_axis[1], z_axis[2],
                           color='deepskyblue', linewidth=2.0, arrow_length_ratio=0.18)
            artists.extend([qx, qy, qz])

        ax.set_xlim(*x_lim); ax.set_ylim(*y_lim); ax.set_zlim(*z_lim)

    return draw_frame, artists


# ═════════════════════════════════════════════════════════════════════════════
# Main visualisation  — side-by-side: Original (left) vs Corrected (right)
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='3D face landmark viewer with head-tilt normalisation')
    parser.add_argument('jsonl', nargs='?', default=None,
                        help='Path to a .jsonl file (interactive picker if omitted)')
    parser.add_argument('--method',
                        choices=['rodrigues', 'procrustes', 'basis',
                                 'affine', 'perspective', 'euler', 'planes',
                                 'frontalize'],
                        default='planes',
                        help='Tilt-correction method (default: planes)')
    args = parser.parse_args()

    jsonl_path = pick_jsonl([args.jsonl] if args.jsonl else [])
    print(f'Loading {jsonl_path} …')
    times, landmarks_raw = load_jsonl(jsonl_path)
    if not times:
        print('No valid frames found.')
        return

    num_frames = len(times)
    t0 = times[0]
    rel_times = [t - t0 for t in times]
    basename = Path(jsonl_path).name
    method = args.method
    print(f'{basename}: {num_frames} frames, {rel_times[-1]:.1f} s total')
    print(f'Tilt correction: {method}')

    # ── Centre on nose tip ───────────────────────────────────────────────
    nose_idx = Mesh_annotation["noseTip"][0]
    for i in range(num_frames):
        landmarks_raw[i] = landmarks_raw[i] - landmarks_raw[i][nose_idx]

    # Keep a deep copy for the uncorrected (original) view
    import copy
    landmarks_orig = copy.deepcopy(landmarks_raw)

    # ── Prepare corrected frame containers ───────────────────────────────
    landmarks_corr = copy.deepcopy(landmarks_raw)
    landmarks_aff = copy.deepcopy(landmarks_raw)
    landmarks_rigid = copy.deepcopy(landmarks_raw)
    euler_angles = []  # (pitch, yaw, roll) per frame
    plane_angles = []  # (pitch, yaw, roll) from plane method
    plane_infos_orig = []
    axis_mats_corr = []
    for i in range(num_frames):
        R, center = estimate_rotation_matrix(landmarks_corr[i])
        euler_angles.append(rotation_to_euler(R, degrees=True))
        if method == 'planes':
            plane_infos_orig.append(measure_angles_from_planes(landmarks_orig[i]))

    if method == 'rodrigues':
        for i in range(num_frames):
            landmarks_corr[i] = normalize_tilt_rodrigues(landmarks_corr[i])
    elif method == 'procrustes':
        ref = landmarks_corr[0].copy()
        for i in range(num_frames):
            landmarks_corr[i] = normalize_tilt_procrustes(landmarks_corr[i], ref)
    elif method == 'basis':
        for i in range(num_frames):
            landmarks_corr[i] = normalize_tilt_basis(landmarks_corr[i])
    elif method in ('affine', 'perspective'):
        for i in range(num_frames):
            R, center = estimate_rotation_matrix(landmarks_corr[i])
            landmarks_corr[i] = normalize_landmarks_3d(landmarks_corr[i], R, center)
    elif method == 'euler':
        for i in range(num_frames):
            p, y, r = euler_angles[i]
            # Negate each angle → build exact inverse rotation
            # R_corr = Rx(-pitch) @ Ry(-yaw) @ Rz(-roll)  ==  R_dev.T
            R_corr = euler_correction_matrix(p, y, r, degrees=True)
            center = (landmarks_corr[i][TOP_IDX] + landmarks_corr[i][BOT_IDX]) / 2.0
            pts_c = landmarks_corr[i] - center
            landmarks_corr[i] = (R_corr @ pts_c.T).T + center
    elif method == 'planes':
        for i in range(num_frames):
            corrected, _R_corr, (p, y, r), _nose_dist = correct_landmarks_from_planes(
                landmarks_corr[i]
            )
            landmarks_corr[i] = corrected
            plane_angles.append((p, y, r))

        # Build per-frame axis matrices from corrected pose for overlay.
        for i in range(num_frames):
            info_corr = measure_angles_from_planes(landmarks_corr[i])
            axis_mats_corr.append(
                euler_to_rotation_matrix(
                    info_corr['pitch'],
                    info_corr['yaw'],
                    info_corr['roll'],
                    degrees=True,
                )
            )
    elif method == 'frontalize':
        for i in range(num_frames):
            landmarks_aff[i] = _frontalize_landmarks_affine(landmarks_aff[i])
            landmarks_rigid[i] = _frontalize_landmarks_rigid(landmarks_rigid[i])

    if method == 'planes':
        print(f'  Frame 0 Plane angles: pitch={plane_angles[0][0]:+.1f}°  '
              f'yaw={plane_angles[0][1]:+.1f}°  roll={plane_angles[0][2]:+.1f}°')
        print(f'  Frame 0 nose distance to origin: {plane_infos_orig[0]["nose_dist"]:.5f}')
    elif method == 'frontalize':
        print('  Frame 0 frontalize computed: affine + rigid')
    else:
        print(f'  Frame 0 Euler: pitch={euler_angles[0][0]:+.1f}°  '
              f'yaw={euler_angles[0][1]:+.1f}°  roll={euler_angles[0][2]:+.1f}°')

    if method == 'frontalize':
        # ── Shared axis limits from original + affine + rigid ───────────
        all_frames = landmarks_orig + landmarks_aff + landmarks_rigid
        x_lim, y_lim, z_lim = _compute_axis_limits(all_frames)

        fig = plt.figure(figsize=(22, 8), facecolor='#1e1e1e')
        ax_left = fig.add_subplot(131, projection='3d', facecolor='#1e1e1e')
        ax_mid = fig.add_subplot(132, projection='3d', facecolor='#1e1e1e')
        ax_right = fig.add_subplot(133, projection='3d', facecolor='#1e1e1e')
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.15, top=0.90,
                            wspace=0.05)

        for ax in (ax_left, ax_mid, ax_right):
            _style_3d_axis(ax)

        ax_left.set_title('Original (no correction)', color='#ff6666', fontsize=12, pad=10)
        ax_mid.set_title('Frontalized [affine]', color='#66ff66', fontsize=12, pad=10)
        ax_right.set_title('Frontalized [rigid]', color='#66ddff', fontsize=12, pad=10)

        draw_left, _ = _make_draw_frame(ax_left, landmarks_orig, x_lim, y_lim, z_lim)
        draw_mid, _ = _make_draw_frame(ax_mid, landmarks_aff, x_lim, y_lim, z_lim)
        draw_right, _ = _make_draw_frame(ax_right, landmarks_rigid, x_lim, y_lim, z_lim)

        draw_left(0)
        draw_mid(0)
        draw_right(0)

        e = euler_angles[0]
        info_title = fig.suptitle(
            f'{basename}   t = 0.000 s  (frame 0 / {num_frames})'
            f'   P={e[0]:+.1f}° Y={e[1]:+.1f}° R={e[2]:+.1f}°  [frontalize]',
            color='white', fontsize=13, y=0.97)
    else:
        # ── Shared axis limits from both views ───────────────────────────
        all_frames = landmarks_orig + landmarks_corr
        x_lim, y_lim, z_lim = _compute_axis_limits(all_frames)

        # ── Figure: two 3D subplots side-by-side ─────────────────────────
        fig = plt.figure(figsize=(18, 8), facecolor='#1e1e1e')
        ax_left = fig.add_subplot(121, projection='3d', facecolor='#1e1e1e')
        ax_right = fig.add_subplot(122, projection='3d', facecolor='#1e1e1e')
        plt.subplots_adjust(left=0.03, right=0.97, bottom=0.15, top=0.90,
                            wspace=0.08)

        _style_3d_axis(ax_left)
        _style_3d_axis(ax_right)
        ax_left.set_title('Original (no correction)', color='#ff6666',
                          fontsize=12, pad=10)
        ax_right.set_title(f'Corrected  [{method}]', color='#66ff66',
                           fontsize=12, pad=10)

        draw_left, _ = _make_draw_frame(ax_left, landmarks_orig,
                                        x_lim, y_lim, z_lim,
                                        plane_infos=plane_infos_orig if method == 'planes' else None)
        draw_right, _ = _make_draw_frame(ax_right, landmarks_corr,
                                         x_lim, y_lim, z_lim,
                                         axis_mats=axis_mats_corr if method == 'planes' else None)

        draw_left(0)
        draw_right(0)

        if method == 'planes':
            p0, y0, r0 = plane_angles[0]
            info_title = fig.suptitle(
                f'{basename}   t = 0.000 s  (frame 0 / {num_frames})'
                f'   P={p0:+.1f}° Y={y0:+.1f}° R={r0:+.1f}°  [planes]',
                color='white', fontsize=13, y=0.97)
        else:
            e = euler_angles[0]
            info_title = fig.suptitle(
                f'{basename}   t = 0.000 s  (frame 0 / {num_frames})'
                f'   P={e[0]:+.1f}° Y={e[1]:+.1f}° R={e[2]:+.1f}°',
                color='white', fontsize=13, y=0.97)

    # ── Slider ───────────────────────────────────────────────────────────
    ax_slider = plt.axes([0.12, 0.04, 0.76, 0.03], facecolor='#333333')
    slider = Slider(ax_slider, 'Time (s)',
                    rel_times[0], rel_times[-1],
                    valinit=0, valstep=None, color='#00aaff')
    ax_slider.xaxis.label.set_color('white')
    for tl in ax_slider.get_xticklabels():
        tl.set_color('white')
    slider.label.set_color('white')
    slider.valtext.set_color('white')

    def update(val):
        fi = int(np.argmin(np.abs(np.array(rel_times) - slider.val)))
        if method == 'frontalize':
            draw_left(fi)
            draw_mid(fi)
            draw_right(fi)
            e = euler_angles[fi]
            info_title.set_text(
                f'{basename}   t = {rel_times[fi]:.3f} s  (frame {fi} / {num_frames})'
                f'   P={e[0]:+.1f}° Y={e[1]:+.1f}° R={e[2]:+.1f}°  [frontalize]')
        else:
            draw_left(fi)
            draw_right(fi)
            if method == 'planes':
                p, y, r = plane_angles[fi]
                info_title.set_text(
                    f'{basename}   t = {rel_times[fi]:.3f} s  (frame {fi} / {num_frames})'
                    f'   P={p:+.1f}° Y={y:+.1f}° R={r:+.1f}°  [planes]')
            else:
                e = euler_angles[fi]
                info_title.set_text(
                    f'{basename}   t = {rel_times[fi]:.3f} s  (frame {fi} / {num_frames})'
                    f'   P={e[0]:+.1f}° Y={e[1]:+.1f}° R={e[2]:+.1f}°')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    if method == 'frontalize':
        ax_left.view_init(elev=-90, azim=90)
        ax_mid.view_init(elev=-90, azim=90)
        ax_right.view_init(elev=-90, azim=90)
    else:
        ax_left.view_init(elev=-90, azim=90)
        ax_right.view_init(elev=-90, azim=90)
    plt.show()


if __name__ == '__main__':
    main()