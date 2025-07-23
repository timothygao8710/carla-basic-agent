#!/usr/bin/env python3
# projection_utils.py
# Frame‑correct, vectorised pin‑hole projection for CARLA (X fwd, Y right, Z up)

from __future__ import annotations
import numpy as np


# ─────────────────────────── intrinsics ────────────────────────────
def get_camera_intrinsics(w: int, h: int, fov_deg: float) -> np.ndarray:
    """Return 3 × 3 pin‑hole intrinsic matrix (same as CARLA/UE4)."""
    f = w / (2.0 * np.tan(np.radians(fov_deg) * 0.5))
    K = np.array([[f, 0, w / 2.0],
                  [0, f, h / 2.0],
                  [0, 0,       1]], dtype=np.float32)
    return K


# ─────────────────────────── projection ────────────────────────────
def project_points(points_world: np.ndarray,
                   world2cam: np.ndarray,
                   K: np.ndarray,
                   front_thresh: float = -1e9) -> np.ndarray:
    if len(points_world) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # homogeneous → camera frame (X fwd, Y right, Z up)
    pts_cam = (world2cam @
               np.hstack([points_world, np.ones((len(points_world), 1))]).T).T[:, :3]

    in_front = pts_cam[:, 0] > front_thresh
    pts_good = pts_cam[in_front]

    if pts_good.size == 0:                     # nothing visible this frame
        return np.full((len(points_world), 2), np.nan, dtype=np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx *  pts_good[:, 1] / pts_good[:, 0] + cx      # Y / X
    v = fy * -pts_good[:, 2] / pts_good[:, 0] + cy      # -Z / X

    pix = np.full((len(points_world), 2), np.nan, dtype=np.float32)
    pix[in_front] = np.stack([u, v], axis=1)
    return pix
