from __future__ import annotations
import argparse, queue, sys
from pathlib import Path
from math import cos, sin, radians
import math
import carla, cv2

from deltas_controller import LearnedLocalPlanner

# --- camera geometry -------------------------------------------------------
import numpy as np

def build_K(w: int, h: int, fov_deg: float) -> np.ndarray:
    """Intrinsic matrix from image size and horizontal FOV (deg)."""
    f = w / (2.0 * np.tan(np.radians(fov_deg) * 0.5))
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0,   1.0 ]])

def get_world2cam(transform: carla.Transform) -> np.ndarray:
    """
    4×4 matrix that converts world coordinates (homogeneous) to the
    left‑handed CARLA camera space (x = forward, y = right, z = up).
    """
    loc   = transform.location
    roll  = np.radians(transform.rotation.roll)
    pitch = np.radians(transform.rotation.pitch)
    yaw   = np.radians(transform.rotation.yaw)

    c_y, s_y = np.cos(yaw),   np.sin(yaw)
    c_r, s_r = np.cos(roll),  np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch)

    # world → camera rotation (R_cam_world)
    R = np.array([
        [ c_p * c_y,                     c_p * s_y,                    -s_p],
        [ s_r * s_p * c_y - c_r * s_y,  s_r * s_p * s_y + c_r * c_y,  s_r * c_p],
        [ c_r * s_p * c_y + s_r * s_y,  c_r * s_p * s_y - s_r * c_y,  c_r * c_p]
    ])
    t = np.array([[loc.x], [loc.y], [loc.z]])
    Rt = np.hstack((R, -R @ t))       # 3×4
    return np.vstack((Rt, [0, 0, 0, 1]))   # 4×4


def init_video(path: str, size: tuple[int, int], fps: int):
    """Open a VideoWriter with a codec that exists on this system."""
    for codec in ("avc1", "mp4v", "MJPG"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if vw.isOpened():
            print(f"[INFO] Recording → {path} (codec {codec})")
            return vw
    print("[WARN] Video disabled – no codec opened")
    return None


def yaw_to_vec(yaw_deg: float) -> carla.Vector3D:
    """Ground-plane forward unit-vector from a yaw angle (deg)."""
    return carla.Vector3D(cos(radians(yaw_deg)), sin(radians(yaw_deg)), 0.0)

def main():
    ap = argparse.ArgumentParser("Δ-Waypoint driver")
    ap.add_argument("--deltas", default="deltas.npy",
                    help="N×2 NumPy file: [Δspeed (m/s), Δyaw (deg)] per tick")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--map",  default="Town05")
    ap.add_argument("--fps",  type=int, default=10)
    ap.add_argument("--video", default="delta_run.mp4")
    ap.add_argument("--show_wp", action="store_true",
                    help="draw target waypoint every tick")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world  = client.load_world(args.map)

    dt = 1.0 / args.fps
    settings              = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = dt
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    spawn_tf = world.get_map().get_spawn_points()[0]
    x0, y0, z0 = spawn_tf.location.x, spawn_tf.location.y, spawn_tf.location.z
    yaw0       = spawn_tf.rotation.yaw

    deltas_path = Path(args.deltas).expanduser()
    if not deltas_path.is_file():
        sys.exit(f"[ERROR] Cannot find {deltas_path}")

    deltas = np.load(deltas_path)
    if deltas.ndim != 2 or deltas.shape[1] != 2:
        sys.exit("[ERROR] Δ file must be N×2: [Δspeed, Δyaw]")

    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.filter("vehicle.*model3")[0]
    ego    = world.spawn_actor(ego_bp, spawn_tf)

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "100")
    
    cam_width, cam_height, cam_fov = 1280, 720, 100
    K  = build_K(cam_width, cam_height, cam_fov)
    Ki = K[:3, :3]                    # 3×3, handy later

    # cam_tf  = carla.Transform(carla.Location(x=1.6, z=1.4))
    
    cam_tf = carla.Transform(
    carla.Location(x=-8.0,   # 8 m behind the rear axle
                   y=0.0,    # centred
                   z=4.0),   # 4 m above ground
    carla.Rotation(pitch=-15,  # slight downward tilt
                   yaw=0.0,
                   roll=0.0))
    
    
    cam     = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    img_q: "queue.Queue[carla.Image]" = queue.Queue(2)
    cam.listen(lambda img: img_q.put(img) if not img_q.full() else None)

    video = init_video(args.video, (1280, 720), args.fps)
    
    print("inits done")
    
    planner = LearnedLocalPlanner(
        ego,
        opt_dict={
            "dt": dt,          # keep controller’s dt in sync with the world
            "target_speed": 0  # will be overwritten every tick anyway
        }
    )

    tick   = 0
    n_d    = len(deltas)
    horizon = 10                  # send a 1‑second horizon each tick

    try:
        while tick < n_d:
            chunk = deltas[tick : tick + horizon]
            
            if chunk.shape[0] < horizon:
                pad = np.zeros((horizon - chunk.shape[0], 2), dtype=np.float32)
                chunk = np.vstack((chunk, pad))

            print(chunk)

            planner.set_relative_trajectory(chunk.tolist(), dt=dt)
            control = planner.run_step(debug=True)
            ego.apply_control(control)
            world.tick()

            try:
                img = img_q.get_nowait()
                frame = np.frombuffer(img.raw_data, np.uint8).reshape(
                    (img.height, img.width, 4)
                )[:, :, :3]

                wps = [wp for wp, _ in list(planner._waypoints_queue)[:10]]

                w2c = get_world2cam(cam.get_transform())
                for wp in wps:
                    p_world = np.array([wp.transform.location.x,
                                        wp.transform.location.y,
                                        wp.transform.location.z,
                                        1.0])
                    p_cam = w2c @ p_world
                    if p_cam[0] <= 0:          # behind the camera
                        continue
                    proj = Ki @ (p_cam[:3] / p_cam[0])
                    u, v = int(proj[0]), int(proj[1])
                    if 0 <= u < cam_width and 0 <= v < cam_height:
                        cv2.circle(frame, (u, v), 6, (0, 0, 255), -1)

                if video:
                    video.write(frame)
            except queue.Empty:
                pass


            tick += 1
            
            print(tick, n_d)

    finally:
        if video:
            video.release()
            print(f"[INFO] Video saved → {args.video}")


    cam.stop(); cam.destroy(); ego.destroy()
    tm.set_synchronous_mode(False)
    settings.synchronous_mode = False
    world.apply_settings(settings)

if __name__ == "__main__":
    main()
