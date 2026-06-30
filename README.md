# ros2_mycobot — MyCobot Pro 630 description + cuRobo motion planner

ROS 2 package(s) for the **MyCobot Pro 630** (6-DOF + suction): the robot **URDF /
description** and the **cuRobo** collision-free / dynamics-aware **motion-planning server**
that the rest of the stack talks to over a socket. The actual robot control and ROS 2 bridge
live in the sibling repos (`../mycobot_mpc`, `../../pick_and_place`).

## Layout

```
src/mycobot_description/
  urdf, meshes, launch …                 # MyCobot Pro 630 URDF (suction-cup TCP = 0.145 m)
  curobo/
    curobo_planner_server.py             # v1 planner server  (curobo env, 0.7.7)
    curobo_planner_server_v2.py          # V2 planner server  (curobo2 env, 0.8.0, dynamics-aware)
    mycobot_pro_630.yml / *_v2.yml        # cuRobo robot config (auto-ported v1 → V2 at startup)
    compute_inertia.py                    # mesh-derived <inertial> tags for the URDF
    generate_spheres.py / viser_spheres_v2.py   # collision-sphere authoring / viz
    mock_planner.py                       # socket-compatible stub (no GPU) for dry runs
```

## Planner server (socket RPC, port 9997)

Both servers speak the same newline-JSON protocol on `127.0.0.1:9997` (so the desktop
controller/ planner nodes are backend-agnostic):

| request `type` | does |
|---|---|
| `plan_pose` / `plan_joint` | collision-free trajectory to a Cartesian pose / joint goal |
| `fk` | forward kinematics (TCP pose for a joint config) |
| `attach` / `detach` | attach a grasped object's bbox to the planner (carried-object collision) |
| `set_world` / `clear_world` | set/reset world obstacle cuboids |
| `ping` | joint names / liveness |

```bash
# V2 (dynamics-aware: B-spline + inverse dynamics + torque limits); curobo2 env
conda activate curobo2
python src/mycobot_description/curobo/curobo_planner_server_v2.py     # 127.0.0.1:9997
```

- **V2** reads link mass/inertia/CoM from the URDF `<inertial>` (mesh-derived via
  `compute_inertia.py`) and joint torque limits from `<limit effort=…>`, and returns the
  **B-spline control points** alongside the sampled trajectory.
- **Safety keep-out:** the V2 base world includes a persistent **`keepout_xneg`** wall (its +x
  face at **x = −0.3**) so no plan ever penetrates that plane; it survives `clear_world` and is
  re-added on every `set_world`.

Consumers: `../mycobot_mpc/curobo_controller_node.py` (goal → trajectory → bridge) and
`../../pick_and_place` (`online_planner_node.py` streams 0.4 s chunks; `servo_touch.py`,
`real_pipeline.py` use `plan_pose`/`fk`/`attach`).
