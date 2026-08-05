# Paste the full contents of your local kp_detection.py here
import numpy as np
from skan import Skeleton

def calculate_deviation_angle(p1, p2, p_mid):
    x1, y1 = p1
    xn, yn = p2
    x_m, y_m = p_mid

    dx_total = xn - x1
    dy_total = yn - y1
    chord_len = np.hypot(dx_total, dy_total)

    if chord_len < 1e-6:
        dist = np.hypot(x_m - x1, y_m - y1)
        return dist, 0.0

    dist = abs(dy_total * x_m - dx_total * y_m + xn * y1 - yn * x1) / chord_len
    base_angle = np.degrees(np.arctan2(dy_total, dx_total))
    seg_angle = np.degrees(np.arctan2(y_m - y1, x_m - x1))

    diff = abs(seg_angle - base_angle)
    diff = min(diff, 360.0 - diff)
    return dist, diff

def detect_keypoints_recursive(stroke, dist_thresh_px=3.0, angle_thresh_deg=25.0):
    if len(stroke) <= 2:
        return [stroke[0], stroke[-1]]

    p1 = stroke[0]
    pn = stroke[-1]

    best_score = -np.inf
    best_idx = -1

    for i in range(1, len(stroke) - 1):
        dist, angle_diff = calculate_deviation_angle(p1, pn, stroke[i])
        dist_norm = dist / max(dist_thresh_px, 1e-6)
        angle_norm = angle_diff / max(angle_thresh_deg, 1e-6)
        joint_score = dist_norm + angle_norm

        if joint_score > best_score:
            best_score = joint_score
            best_idx = i

    if best_idx != -1 and best_score >= 1.0:
        left_kps = detect_keypoints_recursive(stroke[:best_idx + 1], dist_thresh_px, angle_thresh_deg)
        right_kps = detect_keypoints_recursive(stroke[best_idx:], dist_thresh_px, angle_thresh_deg)
        return left_kps[:-1] + right_kps

    return [p1, pn]

def detect_keypoints(skeleton_img, dist_thresh_px=3.0, angle_thresh_deg=25.0):
    bool_skel = (skeleton_img > 0) if np.max(skeleton_img) > 1 else (skeleton_img == 1)

    if bool_skel.sum() < 3:
        return []

    try:
        skel_obj = Skeleton(bool_skel)
    except Exception:
        return []

    raw_kps = []
    for path_idx in range(skel_obj.n_paths):
        coords = skel_obj.path_coordinates(path_idx)
        stroke = [(int(c[1]), int(c[0])) for c in coords]
        if len(stroke) >= 2:
            stroke_kps = detect_keypoints_recursive(stroke, dist_thresh_px, angle_thresh_deg)
            raw_kps.extend(stroke_kps)

    unique_kps = []
    for kp in raw_kps:
        if not any(np.hypot(kp[0] - u[0], kp[1] - u[1]) < 3.5 for u in unique_kps):
            unique_kps.append(kp)

    return unique_kps
