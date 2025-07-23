from __future__ import annotations

import argparse
import math
import queue
import sys
from pathlib import Path

import carla
import cv2
import numpy as np

from deltas_controller import LearnedLocalPlanner
from projection_utils import get_camera_intrinsics, project_points   # ★ NEW

# ───────────────────────────── helpers ────────────────────────────────────
def init_video(path: str, size: tuple[int, int], fps: int) -> cv2.VideoWriter | None:
    """Open a VideoWriter with a codec that exists on this system."""
    for codec in ("avc1", "mp4v", "MJPG"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if vw.isOpened():
            print(f"[INFO] Recording → {path} (codec {codec})")
            return vw
    print("[WARN] Video disabled – no codec opened")
    return None


# ─────────────────────────────── main ─────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser("Δ‑Waypoint driver")
    ap.add_argument("--deltas", default="deltas.npy",
                    help="N×2 NumPy file: [Δspeed (m/s), Δyaw (deg)] per tick")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--map", default="Town05")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--video", default="delta_run.mp4")
    ap.add_argument("--show_wp", action="store_true",
                    help="draw next 10 way‑points as red dots")
    args = ap.parse_args()

    # ── connect & configure world ─────────────────────────────────────────
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.load_world(args.map)

    dt = 1.0 / args.fps
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    # ── load Δ file ───────────────────────────────────────────────────────
    deltas_path = Path(args.deltas).expanduser()
    if not deltas_path.is_file():
        sys.exit(f"[ERROR] Cannot find {deltas_path}")
    deltas = np.load(deltas_path)
    if deltas.ndim != 2 or deltas.shape[1] != 2:
        sys.exit("[ERROR] Δ file must be N×2: [Δspeed, Δyaw]")

    # ── spawn ego vehicle ────────────────────────────────────────────────
    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.filter("vehicle.*model3")[0]
    spawn_tf = world.get_map().get_spawn_points()[0]
    ego = world.spawn_actor(ego_bp, spawn_tf)

    # ── RGB camera (front‑facing) ────────────────────────────────────────
    cam_width, cam_height, cam_fov = 1280, 720, 100
    K = get_camera_intrinsics(cam_width, cam_height, cam_fov)

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(cam_width))
    cam_bp.set_attribute("image_size_y", str(cam_height))
    cam_bp.set_attribute("fov", str(cam_fov))

    # cam_tf = carla.Transform(
    #     carla.Location(x=1.6, z=1.4),   # bonnet‑level, front of the car
    #     carla.Rotation(pitch=0)         # looking straight ahead
    # )
    
    cam_tf = carla.Transform(
    carla.Location(x=-8.0,   # 8 m behind the rear axle
                   y=0.0,    # centred
                   z=4.0),   # 4 m above ground
    carla.Rotation(pitch=-15,  # slight downward tilt
                   yaw=0.0,
                   roll=0.0))
    
    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    img_q: "queue.Queue[carla.Image]" = queue.Queue(2)
    cam.listen(lambda img: img_q.put(img) if not img_q.full() else None)

    video = init_video(args.video, (cam_width, cam_height), args.fps)

    # ── initialise controller ────────────────────────────────────────────
    planner = LearnedLocalPlanner(
        ego,
        opt_dict={
            "dt": dt,
            "target_speed": 0
        }
    )

    n_d = len(deltas)
    horizon = 10
    tick = 0

    print("[INFO] Replay started …")
    try:
        while tick < n_d:
            # horizon slice (zero‑pad near end)
            chunk = deltas[tick: tick + horizon]
            if chunk.shape[0] < horizon:
                chunk = np.vstack((chunk,
                                   np.zeros((horizon - chunk.shape[0], 2),
                                            dtype=np.float32)))
            planner.set_relative_trajectory(chunk.tolist(), dt=dt)

            control = planner.run_step(debug=False)
            ego.apply_control(control)
            world.tick()

            # ── overlay & save frame ────────────────────────────────────
            try:
                img = img_q.get_nowait()
                frame = np.frombuffer(img.raw_data, np.uint8).reshape(
                    (img.height, img.width, 4))[:, :, :3]
                frame = frame.copy()

                # --------------------------------------------------------------------
                # Overlay future way‑points (vectorised & accurate)
                # --------------------------------------------------------------------
                if args.show_wp:
                    world2cam = np.array(cam.get_transform().get_inverse_matrix(),
                                         dtype=np.float32)

                    # project planner waypoints
                    wps = [wp for wp, _ in list(planner._waypoints_queue)]
                    pts_world = np.array([[wp.transform.location.x,
                                           wp.transform.location.y,
                                           wp.transform.location.z]
                                          for wp in wps], dtype=np.float32)

                    uv = project_points(pts_world, world2cam, K)

                    for u, v in uv:
                        if not np.isnan(u):
                            cv2.circle(frame, (int(u), int(v)),
                                       5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                            
                    # project proposed waypoints
                    # print(planner.proposed_wp_list)
                    # print(len(planner.proposed_wp_list))
                    pts_world = np.array(planner.proposed_wp_list, dtype=np.float32)
                    uv = project_points(pts_world, world2cam, K)
                    
                    # temp =0
                    for u, v in uv:
                        if not np.isnan(u):
                            # temp += 1
                            cv2.circle(frame, (int(u), int(v)),
                                       5, (255, 0, 0), -1, lineType=cv2.LINE_AA)
                            
                    # print(temp)

                if video:
                    video.write(frame)
            except queue.Empty:
                pass

            tick += 1

    finally:
        if video:
            video.release()
            print(f"[INFO] Video saved → {args.video}")

        cam.stop()
        cam.destroy()
        ego.destroy()
        tm.set_synchronous_mode(False)
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("[INFO] Replay finished.")


if __name__ == "__main__":
    main()
