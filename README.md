# carla-basic-agent

Minimal implementation of the Planner and VehiclePIDController using CARLA.

## Installation
- CARLA 0.9.15 [[CARLA_0.9.15.tar.gz](https://tiny.carla.org/carla-0-9-15-linux), [AdditionalMaps_0.9.15.tar.gz](https://tiny.carla.org/additional-maps-0-9-15-linux)]
- scenario_runner-0.9.15 [[v0.9.15](https://github.com/carla-simulator/scenario_runner/releases/tag/v0.9.15)]

```bash
apt-get install libomp5

pip install numpy six xmlschema networkx py_trees pygame
cd /home/admin/Programs/scenario_runner-0.9.15
pip install -r requirements.txt # Or run `conda env create -f env.yml` in this directory.
```
Add the following lines to your `~/.bashrc`. Example:
```bash
export CARLA_ROOT=${HOME}/Apps/CARLA_0.9.15/
export SCENARIO_RUNNER_ROOT=${HOME}/Programs/scenario_runner-0.9.15/
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
```

## Method

The system follows a sequential flow of components:

1. **BasicAgent**: Once the `BasicAgent` sets a destination [here](https://github.com/Jiankai-Sun/carla-basic-agent/blob/main/basic_agent.py#L141-L162), the **Global Planner**  generates the shortest path and passes it to the local planner [here](https://github.com/Jiankai-Sun/carla-basic-agent/blob/main/basic_agent.py#L178-L187).
2. **Planner**: The **Global Planner** creates a series of waypoints based on the destination and environment data [here](https://github.com/Jiankai-Sun/carla-basic-agent/blob/main/local_planner.py#L192C9-L217).
3. **VehiclePIDController**: The **VehiclePIDController** then converts the waypoints and target speed (20 km/h) into control signals (steering, throttle, etc.), which are used to navigate the vehicle. See an example of this process [here](https://github.com/carla-simulator/scenario_runner/blob/34e751d3dbf0db95e0808fcd960dc9432df58029/srunner/tests/carla_mocks/agents/navigation/controller.py#L54-L92).


## Run
```bash
python automatic_control.py --agent Basic
```

cd /home/timothygao/VLA_driving/software && ./CarlaUE4.sh -RenderOffScreen -carla-server -benchmark -fps=10

## Acknowledgment
- [modulardecision](https://github.com/decisionforce/modulardecision)
- [CARLA scenario_runner-0.9.15](https://github.com/carla-simulator/scenario_runner/releases/tag/v0.9.15)
