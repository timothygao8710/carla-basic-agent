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
from PIL import Image
import requests
from projection_utils import get_camera_intrinsics, project_points
import io, base64
def init_video(path: str, size: tuple[int, int], fps: int) -> cv2.VideoWriter | None:
    """Open a VideoWriter with a codec that exists on this system."""
    for codec in ("avc1", "mp4v", "MJPG"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if vw.isOpened():
            print(f"[INFO] Recording → {path} (codec {codec})")
            return vw
    print("[WARN] Video disabled – no codec opened")
    return None
def _encode_png_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
################################################################################################################################################################


#### SETTINGS ####

# model settings
VLA_SERVER = 'http://127.0.0.1:8080/gen_action'
ACTION_HORIZON = 6 # change according to model, 3 * model_FPS

# script settings
FPS = 10
DO_SHOW_WP = True # wether to plot the red/blue dots
max_drive_len = FPS * 60 * 5 # 5 minutess
SAVE_VIDEO_FILE = '/home/timothygao/carla-basic-agent/closed_loop_test.mp4'

# carla settings
MAP = 'Town05'
SPAWN_POINT = 0
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
#################

def get_deltas(current_view: Image.Image, user_instruction: str, current_speed: float):
    # FOR NOW, DIRECTLY QUERY VLA WITH THIS
    
    # TODO: GEMINI LOGIC HERE
    
    res = send_to_vla_server(pil_image=current_view, prompt=user_instruction, proprio=current_speed)
    return res

def get_vehicle_state(vehicle):
    v = vehicle.get_velocity() # 3-D vector of m/s in x,y,z
    speed = (v.x**2 + v.y**2 + v.z**2) ** 0.5 
    yaw   = vehicle.get_transform().rotation.yaw # in degrees
    return speed, yaw

def send_to_vla_server(pil_image: Image.Image, prompt: str, proprio: float) -> np.ndarray:
    # handles all the batching stuff
    payload = {
        "obs_images":     [_encode_png_b64(pil_image)], # batch size 1
        "prompts":  [prompt],
        "proprios": [[proprio]],
    }

    try:
        r = requests.post(VLA_SERVER, json=payload, timeout=300.0)
        
        if(r.status_code != 200):
            print("[WARN] VLA HTTP error", r.status_code)
            assert(False)
        
        res = np.array(r.json()["action"], dtype=np.float32)
        print(f"model output shape {res.shape}")
        return res[0] # UNBATCH
    except Exception as e:
        print("[WARN] VLA request failed:", e)
        return np.zeros((ACTION_HORIZON, 2), dtype=np.float32)
    
    
def main() -> None:
    # ── connect & configure world ─────────────────────────────────────────
    print("Configuring world...")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.load_world(MAP)

    dt = 1.0 / FPS
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    
    print("World configured")

    # ── spawn ego vehicle ────────────────────────────────────────────────
    print("Spawning vehicles ...")
    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.filter("vehicle.*model3")[0]
    spawn_tf = world.get_map().get_spawn_points()[SPAWN_POINT]
    ego = world.spawn_actor(ego_bp, spawn_tf)
    print("Vehicles spawned")

    # ── RGB camera (front‑facing) ────────────────────────────────────────
    
    print("Spawning cameras ...")
    cam_width, cam_height, cam_fov = 1280, 720, 100
    K = get_camera_intrinsics(cam_width, cam_height, cam_fov)

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(cam_width))
    cam_bp.set_attribute("image_size_y", str(cam_height))
    cam_bp.set_attribute("fov", str(cam_fov))

    cam_tf = carla.Transform(
        carla.Location(x=1.6, z=1.4),   # bonnet‑level, front of the car
        carla.Rotation(pitch=0)         # looking straight ahead
    )
    
    # cam_tf = carla.Transform(
    # carla.Location(x=-8.0,   # 8 m behind the rear axle
    #                y=0.0,    # centred
    #                z=4.0),   # 4 m above ground
    # carla.Rotation(pitch=-15,  # slight downward tilt
    #                yaw=0.0,
    #                roll=0.0))
    
    cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    img_q: "queue.Queue[carla.Image]" = queue.Queue(2)
    cam.listen(lambda img: img_q.put(img) if not img_q.full() else None)

    video = init_video(SAVE_VIDEO_FILE, (cam_width, cam_height), FPS)

    print("Cameras spawned...")

    print("Starting closed loop")

    planner = LearnedLocalPlanner(
        ego,
        opt_dict={
            "dt": dt,
            "target_speed": 0
        }
    )

    cur_instruction = 'The vehicle drives straight aggressively.'

    try:
        for tick in range(max_drive_len):
            world.tick() # tick before
            
            try:
                img = img_q.get_nowait()
                buf   = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))[:, :, :3]        # BGRA → BGR
                frame = buf.copy()
                
                pil_frame = Image.fromarray(frame[:, :, ::-1]) # BGR -> RGB
                # this pil_frame fed into model - before projecting waypoints on it
                
                
                # FOR NOW: UPDATE LANGUAGE INSTRUCTIONS EVERY ONE SECOND
                
                # if tick % FPS == 0:
                #     cur_instruction = input("Enter language instruction: ")
                
                # run get_deltas
                
                cur_speed, _ = get_vehicle_state(ego)
                
                print(f"{tick} current speed {cur_speed}")
                
                chunk = get_deltas(pil_frame, cur_instruction, cur_speed)
                
                print(f"{tick} chunk: {chunk}")
                
                if chunk.shape[0] < ACTION_HORIZON: # zero pad
                    print("ZERO PADDED")
                    chunk = np.vstack((chunk,
                                    np.zeros((ACTION_HORIZON - chunk.shape[0], 2),
                                                dtype=np.float32)))
                    
                    
                    
                planner.set_relative_trajectory(chunk.tolist(), dt=dt)
                control = planner.run_step(debug=False)
                ego.apply_control(control)
                
                # show waypoints

                if DO_SHOW_WP:
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
                    for u, v in uv:
                        if not np.isnan(u):
                            cv2.circle(frame, (int(u), int(v)),
                                       5, (255, 0, 0), -1, lineType=cv2.LINE_AA)

                # this pil_frame for vis, after projecting waypoints on it
                pil_frame = Image.fromarray(buf[:, :, ::-1]) # BGR -> RGB
                pil_frame.save('/home/timothygao/carla-basic-agent/vis.png')
                print(f" {tick} saved to /home/timothygao/carla-basic-agent/vis.png")

                if video:
                    video.write(frame)
            except queue.Empty:
                print(f"Video queue empty on {tick}")
                pass

    finally: # we always save video even if e.g., crash
        if video:
            video.release()
            print(f"[INFO] Video saved → {SAVE_VIDEO_FILE}")

        cam.stop()
        cam.destroy()
        ego.destroy()
        tm.set_synchronous_mode(False)
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("[INFO] Replay finished.")


if __name__ == "__main__":
    main()