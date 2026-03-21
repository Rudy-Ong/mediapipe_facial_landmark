import sys
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from face_normalize import estimate_rotation_matrix, normalize_landmarks_3d

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

    "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
    "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],

    "noseTip": [1],
    "noseBottom": [2],
    "noseRightCorner": [98],
    "noseLeftCorner": [327],
 }

# Parts to plot and their display colours
PARTS = {
    "silhouette":     ("white", 1.0),
    "lipsUpperOuter": ("red", 1.0),
    "lipsUpperInner": ("red", 1.0),
    "lipsLowerOuter": ("red", 1.0),
    "lipsLowerInner": ("red", 1.0),
    "rightEyeUpper0": ("cyan", 1.0),
    "rightEyeLower0": ("cyan", 1.0),
    "noseTip": ("magenta", 1.0),
    "noseBottom": ("magenta", 1.0),
    "noseRightCorner": ("magenta", 1.0),
    "noseLeftCorner": ("magenta", 1.0),
}

# ── Load JSONL ───────────────────────────────────────────────────────────────
def load_jsonl(path: str):
    """Return (times, landmarks) from a JSONL file.

    times      – list[float]          Unix timestamps
    landmarks  – list[np.ndarray]     each shape (478, 3)  (first face only)
    """
    times = []
    landmarks = []
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
            # Take the first face
            face = lm_faces[0]
            pts = np.array([[p.get('x', 0), p.get('y', 0), p.get('z', 0)]
                            for p in face], dtype='float64')
            times.append(float(t))
            landmarks.append(pts)
    return times, landmarks


# ── Pick the JSONL file ──────────────────────────────────────────────────────
def pick_jsonl(path_arg=None):
    """Return a JSONL path from CLI arg or the most-recent file in output/stream."""
    if path_arg:
        return path_arg
    files = sorted(glob.glob('output/stream/*.jsonl'))
    if not files:
        print('No JSONL files found in output/stream/. '
              'Pass a path as the first argument.')
        sys.exit(1)
    return files[-1]

def select_jsonl(json_path: str):
    """Return a JSONL path from CLI arg or the most-recent file in output/stream."""

    files = sorted(glob.glob('output/stream/*.jsonl'))
    if not files:
        print('No JSONL files found in output/stream/. '
              'Pass a path as the first argument.')
        sys.exit(1)
    return files[-1]

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Visualise streamed face landmarks in 3D')
    parser.add_argument('jsonl', nargs='?', default=None,
                        help='Path to a JSONL file (defaults to latest output/stream/*.jsonl)')
    parser.add_argument('--method', choices=['raw', 'rigid'], default='raw',
                        help='raw = nose-centred landmarks; rigid = canonical front-facing')
    args = parser.parse_args()

    jsonl_path = pick_jsonl(args.jsonl)
    print(f'Loading {jsonl_path} …')
    times, landmarks = load_jsonl(jsonl_path)
    if len(times) == 0:
        print('No valid frames found.')
        return

    num_frames = len(times)
    t0 = times[0]
    # Relative time in seconds from the first frame
    rel_times = [t - t0 for t in times]

    print(f'{num_frames} frames, {rel_times[-1]:.1f} s total')
    print(f'View mode: {args.method}')

    # ── Centre each frame by subtracting the nose tip position ─────────────
    nose_idx = Mesh_annotation["noseTip"][0]  # index 1
    for i in range(num_frames):
        anchor = landmarks[i][nose_idx]       # x, y, z of nose tip
        landmarks[i] = landmarks[i] - anchor  # shift all points
        if args.method == 'rigid':
            R, center = estimate_rotation_matrix(landmarks[i])
            landmarks[i] = normalize_landmarks_3d(landmarks[i], R, center)

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8), facecolor='#1e1e1e')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1e1e1e')
    plt.subplots_adjust(bottom=0.18)

    # Style the axes
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color('white')
        axis.set_pane_color((0.12, 0.12, 0.12, 1.0))
        for tick in axis.get_ticklabels():
            tick.set_color('gray')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Compute global axis limits across ALL frames for stable view
    all_pts = np.vstack(landmarks)  # (N*478, 3)
    # Gather only the indices we actually plot
    used_indices = set()
    for name in PARTS:
        used_indices.update(Mesh_annotation[name])
    used_indices = sorted(used_indices)

    mins = all_pts[used_indices].min(axis=0)
    maxs = all_pts[used_indices].max(axis=0)
    pad = 0.05
    x_lim = (mins[0] - pad, maxs[0] + pad)
    y_lim = (maxs[1] + pad, mins[1] - pad)  # flip Y so face is upright
    z_lim = (mins[2] - pad, maxs[2] + pad)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)

    # ── Drawing helper ───────────────────────────────────────────────────────
    artists = []  # keep references so we can remove them on redraw

    def draw_frame(frame_idx):
        """Clear old artists and draw the landmarks for a single frame."""
        for a in artists:
            a.remove()
        artists.clear()

        fd = landmarks[frame_idx]
        for name, (colour, alpha) in PARTS.items():
            idx = Mesh_annotation[name]
            xs = fd[idx, 0]
            ys = fd[idx, 1]
            zs = fd[idx, 2]
            s = ax.scatter(xs, ys, zs, c=colour, alpha=alpha,
                           s=8, edgecolors='none')
            artists.append(s)
            line, = ax.plot(xs, ys, zs, color=colour,
                            alpha=alpha * 0.7, linewidth=1)
            artists.append(line)

        # Re-apply fixed limits (scatter auto-rescales otherwise)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_zlim(*z_lim)

    # ── Initial plot (frame 0) ───────────────────────────────────────────────
    draw_frame(0)

    title = ax.set_title(f't = 0.000 s  (frame 0 / {num_frames})',
                         color='white', fontsize=12)

    # ── Slider ───────────────────────────────────────────────────────────────
    ax_slider = plt.axes([0.12, 0.05, 0.76, 0.03],
                         facecolor='#333333')
    slider = Slider(ax_slider, 'Time (s)',
                    rel_times[0], rel_times[-1],
                    valinit=0, valstep=None,
                    color='#00aaff')
    ax_slider.xaxis.label.set_color('white')
    for tick_label in ax_slider.get_xticklabels():
        tick_label.set_color('white')
    slider.label.set_color('white')
    slider.valtext.set_color('white')

    def update(val):
        target = slider.val
        # Find the closest frame to the slider time
        frame_idx = int(np.argmin(np.abs(np.array(rel_times) - target)))

        draw_frame(frame_idx)

        title.set_text(f't = {rel_times[frame_idx]:.3f} s  '
                       f'(frame {frame_idx} / {num_frames})')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    ax.view_init(elev=-90, azim=90)
    plt.show()

if __name__ == '__main__':
    main()
