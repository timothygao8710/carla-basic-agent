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
import time
import random
from io import BytesIO
seed = 42
random.seed(seed); np.random.seed(seed)
def init_video(path: str, size: tuple[int, int], fps: int) -> cv2.VideoWriter | None:
    """Open a VideoWriter with a codec that exists on this system."""
    for codec in ("avc1", "mp4v", "MJPG"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if vw.isOpened():
            print(f"[INFO] Recording → {path} (codec {codec})")
            return vw
    print("[WARN] Video disabled – no codec opened")
    return None
# def _encode_png_b64(pil_img: Image.Image) -> str:
#     buf = io.BytesIO()
#     pil_img.save(buf, format="PNG")
#     return base64.b64encode(buf.getvalue()).decode()
def numpy_to_base64(pil_image):
    buffer = BytesIO()
    if pil_image.mode == 'RGBA':
        print("Unexpected")
        pil_image = pil_image.convert('RGB')
    pil_image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str
_VALID_WP = [
    getattr(carla.WeatherParameters, attr)
    for attr in dir(carla.WeatherParameters)
    if not attr.startswith("_")
    and isinstance(getattr(carla.WeatherParameters, attr), carla.WeatherParameters)
]
def random_weather() -> carla.WeatherParameters:
    base = random.choice(_VALID_WP)
    wp = carla.WeatherParameters(
        cloudiness             = base.cloudiness,
        precipitation          = base.precipitation,
        precipitation_deposits = base.precipitation_deposits,
        wind_intensity         = base.wind_intensity,
        fog_density            = base.fog_density,
        fog_distance           = base.fog_distance,
        wetness                = base.wetness,
        fog_falloff            = base.fog_falloff,
        sun_altitude_angle     = random.uniform(-30.0, 90.0),
        sun_azimuth_angle      = random.uniform(0.0, 360.0),
    )
    return wp
STYLE_PRESETS = {
    "cautious":   dict(dist=6.0, speed_delta=+20, lane_chg=15),
    "normal":     dict(dist=3.0, speed_delta=0,   lane_chg=15),
    "aggressive": dict(dist=1.5, speed_delta=-30, lane_chg=15),
}
def apply_style(tm: carla.TrafficManager, actor: carla.Actor, style: str):
    p = STYLE_PRESETS[style]
    tm.distance_to_leading_vehicle(actor, p["dist"])
    tm.vehicle_percentage_speed_difference(actor, p["speed_delta"])
    tm.random_left_lanechange_percentage(actor, p["lane_chg"])
    tm.random_right_lanechange_percentage(actor, p["lane_chg"])
################################################################################################################################################################

def spawn_npc_vehicles(
    world: carla.World,
    tm: carla.TrafficManager,
    ego: carla.Vehicle,
    max_radius : int,
    n_try_spawns: int = 20, # how many to try to spawn
) -> List[carla.Vehicle]:
    # 1) get all spawn-points and filter by distance to ego
    ego_loc = ego.get_location()
    all_spawns = world.get_map().get_spawn_points()
    nearby_spawns = [
        sp for sp in all_spawns
        if sp.location.distance(ego_loc) <= max_radius
    ]
    random.shuffle(nearby_spawns)
    vehicles: List[carla.Vehicle] = []
    bps = world.get_blueprint_library().filter("vehicle.*")

    for sp in nearby_spawns:
        if len(vehicles) >= n_try_spawns:
            break
        veh_bp = random.choice(bps)
        v = world.try_spawn_actor(veh_bp, sp)
        if not v:
            continue

        style = random.choices(
            ["cautious", "normal", "aggressive"],
            weights=[0.3, 0.5, 0.2],
            k=1
        )[0]
        apply_style(tm, v, style)
        v.set_autopilot(True, tm.get_port())

        vehicles.append(v)

    print(f"{len(vehicles)} NPC vehicles (within {max_radius}m of ego)")
    return vehicles

def spawn_walkers(client: carla.Client, world: carla.World,
                  n_min=150, n_max=300) -> List[int]:
    n_min = int(n_min * WALKER_SCALE)
    n_max = int(n_max * WALKER_SCALE)
    walker_bp = world.get_blueprint_library().filter("walker.pedestrian.*")
    spawn_pts = []
    for _ in range(random.randint(n_min, n_max)):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_pts.append(carla.Transform(loc))
    batch, walker_ids, ctrl_ids = [], [], []
    for sp in spawn_pts:
        batch.append(carla.command.SpawnActor(random.choice(walker_bp), sp))
    for r in client.apply_batch_sync(batch, True):
        if r.error:
            continue
        walker_ids.append(r.actor_id)
    batch.clear()
    for wid in walker_ids:
        ctrl_bp = world.get_blueprint_library().find("controller.ai.walker")
        batch.append(carla.command.SpawnActor(ctrl_bp, carla.Transform(), wid))
    for r in client.apply_batch_sync(batch, True):
        if r.error:
            continue
        ctrl_ids.append(r.actor_id)
    for cid in ctrl_ids:
        ctrl = world.get_actor(cid)
        ctrl.start()
        ctrl.set_max_speed(random.uniform(0.7, 2.0))
    print(f"Spawned {len(walker_ids)} pedestrians")
    return walker_ids + ctrl_ids

def spawn_static_props(world, ego,
                               ids=None, n=40, radius=60, rng_seed=42):
    if ids is None:
        ids = [
            "static.prop.streetbarrier",
            "static.prop.constructioncone",
            "static.prop.trafficcone01",
            "static.prop.warningconstruction",
        ]

    rng       = random.Random(rng_seed)
    bps       = [world.get_blueprint_library().find(i) for i in ids]
    ego_loc   = ego.get_location()
    m         = world.get_map()

    # ① dense grid of waypoints every 2 m (positional arg only!)
    pool = [wp for wp in m.generate_waypoints(2.0)        # :contentReference[oaicite:1]{index=1}
            if wp.lane_type & carla.LaneType.Driving
            and wp.transform.location.distance(ego_loc) < radius]

    rng.shuffle(pool)
    spawned = []

    # ② for each candidate, try several heights until one fits
    for wp in pool:
        if len(spawned) >= n:
            break

        base_tf = wp.transform
        base_tf.rotation.yaw += rng.uniform(-180, 180)

        for dz in (0.3, 0.6, 1.0):                    # progressively higher
            tf = carla.Transform(
                    carla.Location(base_tf.location.x,
                                   base_tf.location.y,
                                   base_tf.location.z + dz),
                    base_tf.rotation)

            prop = world.try_spawn_actor(rng.choice(bps), tf)
            if prop:
                prop.set_simulate_physics(True)       # let it drop onto road
                spawned.append(prop)
                break                                 # success → next prop

    print(f"Spawned {len(spawned)} on-road props")
    return spawned




#### SETTINGS ####

# model settings
VLA_SERVER = 'http://127.0.0.1:8080/gen_action'
ACTION_HORIZON = 6 # change according to model, 3 * model_FPS

# script settings
FPS = 10
DO_SHOW_WP = True # wether to plot the red/blue dots
max_drive_len = FPS * 60 * 5 # 5 minutess
SAVE_VIDEO_FILE = f'/home/timothygao/carla-basic-agent/closed_loop_test_{str(time.time())[:10]}.mp4'

# carla settings
MAP = 'Town05'
# MAP = 'Town01_Opt'
SPAWN_POINT = 3
SPAWN_X = None
SPAWN_Y = None # If Spawn x and spawn y are both not None, will spawn at this location over spawn point
SPAWN_AT_INTERSECTION = False

SPAWN_FROM_TRAJ = '000006' # e.g., '000004', '000041', '000234'
SPAWN_FROM_TRAJ_FRAME = 0

LOAD_TRAJ = None # Load spawn location of trajectory
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000

RANDOMIZE_WEATHER = False
SPAWN_NPC_CARS = False
SPAWN_NPC_PEDESTRIANS = False
SPAWN_RADIUS = 1000 # in meters
SPAWN_STATIC_PROPS = False
SPAWN_TRUCK = False
#################

if SPAWN_AT_INTERSECTION:
    SPAWN_X = -51.058743
    SPAWN_Y = -16.55969
    MAP = 'Town05'

if SPAWN_FROM_TRAJ:
    
    loc = f'/scratch/current/timothygao/july_11_v3/{SPAWN_FROM_TRAJ}/metrics.csv'
    import pandas as pd
    df = pd.read_csv(loc)
    SPAWN_X, SPAWN_Y = df.loc[SPAWN_FROM_TRAJ_FRAME, 'x'], df.loc[SPAWN_FROM_TRAJ_FRAME, 'y']
    
    import json
    traj_jsonl = f'/scratch/current/timothygao/july_11_v3/{SPAWN_FROM_TRAJ}/traj.jsonl'
    with open(traj_jsonl, 'r') as f:
        data = json.load(f)
    MAP = data['map']

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
        # "obs_images":     [_encode_png_b64(pil_image)], # batch size 1
        "obs_images": [numpy_to_base64(pil_image)], # batch size 1
        "prompts":  [prompt],
        "proprios": [[proprio]],
    }

    try:
        r = requests.post(VLA_SERVER, json=payload, timeout=300.0)
        
        if(r.status_code != 200):
            print("[WARN] VLA HTTP error", r.status_code)
            assert(False)
        
        res = np.array(r.json()["action"], dtype=np.float32)
        # print("RAW", res)
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
    settings.deterministic_ragdolls = True
    world.apply_settings(settings)
    
    if RANDOMIZE_WEATHER:
        print("Randomizing weather")
        world.set_weather(random_weather())

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)
    world.set_pedestrians_seed(seed) 
    
    print("World configured")

    # ── spawn ego vehicle ────────────────────────────────────────────────
    print("Spawning vehicles ...")
    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.find("vehicle.tesla.model3")
    spawn_tf = world.get_map().get_spawn_points()[SPAWN_POINT]
    
    if SPAWN_X and SPAWN_Y:
        spawn_tf.location.x = SPAWN_X
        spawn_tf.location.y = SPAWN_Y
    
    ego = world.spawn_actor(ego_bp, spawn_tf)
    
    if SPAWN_NPC_CARS:
        vehicles = spawn_npc_vehicles(world, tm, ego, max_radius = SPAWN_RADIUS)
        
    if SPAWN_NPC_PEDESTRIANS:
        walkers  = spawn_walkers(client, world)
        
    if SPAWN_STATIC_PROPS:
        # bp_lib = world.get_blueprint_library()
        # for bp in bp_lib.filter("static.prop.*"):
        #     print(bp.id)
            
        props = spawn_static_props(world, ego, n=10, radius=60)

    if SPAWN_TRUCK:
        
        truck_bp = world.get_blueprint_library().find("vehicle.carlamotors.firetruck")
        block_tf = carla.Transform(
            ego.get_transform().location + carla.Location(x=15, y=0, z=0),
            carla.Rotation(yaw=random.uniform(-180, 180))
        )
        truck = world.try_spawn_actor(truck_bp, block_tf)
        if truck:
            truck.set_autopilot(False)
            truck.apply_control(carla.VehicleControl(hand_brake=True, throttle=0.0))
        
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
            # "dt": dt,
            "dt": 0.5,
            "target_speed": 0
        }
    )

    # cur_instruction = 'The vehicle accelerates and maintains speed while continuing straight aggressively.'
    # cur_instruction = 'The vehicle accelerates and decelerates while continuing straight, cautiously.'
    cur_instruction = 'The vehicle turns left following the bend normally'
    
    first_time = True
    
    for _ in range(FPS * 2):
        world.tick() # settle
        
    try:
        for tick in range(max_drive_len):
            print(ego.get_transform().location)
            
            world.tick() # tick before
            
            try:
                img = img_q.get_nowait()
                buf   = np.frombuffer(img.raw_data, np.uint8).reshape((img.height, img.width, 4))[:, :, :3]        # BGRA → BGR
                frame = buf.copy()
                
                pil_frame = Image.fromarray(frame[:, :, ::-1]) # BGR -> RGB
                # pil_frame.save(f'/home/timothygao/carla-basic-agent/run_frames/{tick}.png')
                # this pil_frame fed into model - before projecting waypoints on it
                
                
                # FOR NOW: UPDATE LANGUAGE INSTRUCTIONS EVERY ONE SECOND
                
                # if tick % FPS == 0:
                #     cur_instruction = input("Enter language instruction: ")
                
                # run get_deltas
                
                
                cur_speed, _ = get_vehicle_state(ego)
                # cur_speed = planner.cur_speed
                
                if tick < 2:
                    cur_speed = max(cur_speed, 5)
                
                print(f"{tick} current speed {cur_speed}")
                
                chunk = get_deltas(pil_frame, cur_instruction, cur_speed)
                
                print(f"{tick} chunk: {chunk}")
                
                # # linear inteporlation
                
                # interp_chunk = []
                # for i in range(6):
                #     for j in range(5):
                #         interp_chunk.append([5 * chunk[i][0] / 5, 5 * chunk[i][1] / 5])
                # chunk = np.array(interp_chunk)
                    
                # planner.set_relative_trajectory(chunk.tolist(), dt=dt)
                planner.set_relative_trajectory(chunk.tolist(), dt=0.5)
                
                control = planner.run_step(debug=False)
                ego.apply_control(control)
                # world.tick()
                
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
                pil_frame = Image.fromarray(frame[:, :, ::-1]) # BGR -> RGB
                # pil_frame.save('/home/timothygao/carla-basic-agent/vis.png')
                pil_frame.save(f'/home/timothygao/carla-basic-agent/run_frames/{tick}.png')
                print(f" {tick} saved to /home/timothygao/carla-basic-agent/run_frames/{tick}.png")

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