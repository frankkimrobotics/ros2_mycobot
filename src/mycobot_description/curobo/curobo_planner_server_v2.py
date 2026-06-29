#!/usr/bin/env python3
"""curobo_planner_server_v2 :: GPU motion-planning server (cuRobo **V2**, v0.8.0).

Drop-in replacement for ``curobo_planner_server.py`` (the v0.7 server): same
newline-JSON TCP protocol, so ``curobo_controller_node.py`` talks to either one
unchanged. Differences vs the v1 server:

  * Uses the cuRobo **V2** API (``MotionPlanner`` / ``MotionPlannerCfg``), whose
    trajectory optimizer is **dynamics-aware** (B-spline parametrization, torque
    limits) and uses **inverse dynamics**. V2's URDF parser reads link
    mass/inertia/CoM from the URDF ``<inertial>`` tags and joint torque limits
    from ``<limit effort=...>`` -- i.e. the inertials we filled now actually feed
    the planner. Run it in the ``curobo2`` conda env.
  * The response additionally carries the **B-spline control points**
    (``control_points``: [n_ctrl][dof] rad) alongside the sampled trajectory.

The robot config is ported from the v1 yaml (mycobot_pro_630.yml) to the V2
schema in-memory at startup (single source of truth), written next to it as
mycobot_pro_630_v2.yml.

Protocol (one JSON object per line; request -> response)
--------------------------------------------------------
    -> {"type":"ping"}
    <- {"ok":true,"dof":6,"joint_names":[...],"tool_frame":"tcp","backend":"curobo_v2"}

    -> {"type":"plan_pose","start_q":[6 rad],
        "goal_pose":[x,y,z, qw,qx,qy,qz], "max_attempts":5}
    -> {"type":"plan_joint","start_q":[6 rad],"goal_q":[6 rad],"max_attempts":5}
    <- {"success":bool,"trajectory":[[6 rad],...],"dt":float,
        "control_points":[[6 rad],...],"motion_time":float,
        "solve_time":float,"status":str}

Run (in the curobo2 env):
    conda activate curobo2
    python curobo_planner_server_v2.py            # 127.0.0.1:9997
"""
import argparse
import json
import os
import socket
import threading
import traceback

import yaml
import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.state.state_joint import JointState
from curobo._src.geom.types import Cuboid, SceneCfg

HERE = os.path.dirname(os.path.abspath(__file__))
V1_CFG = os.path.join(HERE, "mycobot_pro_630.yml")
V2_CFG = os.path.join(HERE, "mycobot_pro_630_v2.yml")


def port_v1_to_v2(v1_path, v2_path):
    """Translate the v0.7 robot_cfg yaml to the V2 schema; write and return path."""
    cfg = yaml.safe_load(open(v1_path))
    kin = cfg["robot_cfg"]["kinematics"]
    kin["format_version"] = 2.0
    if "ee_link" in kin:                      # v1 single ee_link -> v2 tool_frames list
        kin["tool_frames"] = [kin.pop("ee_link")]
    kin.pop("link_names", None)               # v1-only key
    cs = kin["cspace"]
    if "retract_config" in cs:                # v1 retract_config -> v2 default_joint_position
        cs["default_joint_position"] = cs.pop("retract_config")
    with open(v2_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return kin["tool_frames"][0]


def write_world_yaml(path, ground_z):
    """A thin ground slab whose top face sits at ground_z (base_link frame)."""
    world = {}
    if ground_z is not None:
        world["cuboid"] = {
            "ground": {"dims": [2.0, 2.0, 0.04],
                       "pose": [0.0, 0.0, ground_z - 0.02, 1, 0, 0, 0]}
        }
    with open(path, "w") as f:
        yaml.safe_dump(world, f, sort_keys=False)
    return path


class Planner:
    def __init__(self, ground_z=-0.1, world_path=None, interpolation_dt=0.02,
                 collision_activation_distance=0.01, num_trajopt_seeds=4, num_ik_seeds=32):
        self.tool_frame = port_v1_to_v2(V1_CFG, V2_CFG)

        if world_path is None:
            world_path = os.path.join(HERE, "_world_v2.yml")
            write_world_yaml(world_path, ground_z)

        cfg = MotionPlannerCfg.create(
            robot=V2_CFG,
            scene_model=os.path.abspath(world_path),
            use_cuda_graph=False,          # flexible problem shapes (pose + cspace)
            num_trajopt_seeds=num_trajopt_seeds,
            num_ik_seeds=num_ik_seeds,
            optimizer_collision_activation_distance=collision_activation_distance,
            collision_cache={"primitive": 16},   # room for box walls (cuboids)
        )
        self.mp = MotionPlanner(cfg)
        self._lock = threading.Lock()
        self.joint_names = list(self.mp.default_joint_state.joint_names)
        self.dof = len(self.joint_names)
        self.device = self.mp.default_joint_state.position.device

        # base world (ground only) kept so set_world can rebuild ground + boxes
        self._base_cuboids = {}
        if ground_z is not None:
            self._base_cuboids["ground"] = {
                "dims": [2.0, 2.0, 0.04],
                "pose": [0.0, 0.0, ground_z - 0.02, 1, 0, 0, 0]}

        print(f"[planner-v2] warming up MotionPlanner...", flush=True)
        self.mp.warmup()
        print(f"[planner-v2] ready. dof={self.dof} joints={self.joint_names} "
              f"tool_frame={self.tool_frame}", flush=True)

    # ---- helpers ----
    def _start_state(self, q):
        t = torch.tensor([q], dtype=torch.float32, device=self.device)
        return JointState.from_position(t, joint_names=self.joint_names)

    def _result_to_dict(self, res):
        if res is None:
            return {"success": False, "status": "no solution", "trajectory": [],
                    "control_points": [], "dt": 0.0, "motion_time": 0.0, "solve_time": 0.0}
        ok = bool(res.success.any().item())
        out = {"success": ok,
               "status": "success" if ok else "failed",
               "solve_time": float(getattr(res, "solve_time", 0.0) or 0.0)}
        if ok:
            plan = res.get_interpolated_plan()
            pos = plan.position.squeeze(0).squeeze(0)          # [N, dof]
            out["trajectory"] = pos.detach().cpu().numpy().round(6).tolist()
            out["dt"] = float(plan.dt)
            cps = res.solution[0, 0]                            # [n_ctrl, dof] best seed
            out["control_points"] = cps.detach().cpu().numpy().round(6).tolist()
            try:
                out["motion_time"] = float(res.motion_time())
            except Exception:
                out["motion_time"] = float(plan.dt) * pos.shape[0]
        else:
            out.update(trajectory=[], control_points=[], dt=0.0, motion_time=0.0)
        return out

    # ---- planning ----
    def plan_pose(self, start_q, goal_pose, max_attempts=5):
        with self._lock:
            start = self._start_state(start_q)
            p = torch.tensor([goal_pose[0:3]], dtype=torch.float32, device=self.device)
            quat = torch.tensor([goal_pose[3:7]], dtype=torch.float32, device=self.device)  # wxyz
            goal = GoalToolPose.from_poses({self.tool_frame: Pose(position=p, quaternion=quat)})
            res = self.mp.plan_pose(goal, start, max_attempts=max_attempts)
            return self._result_to_dict(res)

    def plan_joint(self, start_q, goal_q, max_attempts=5):
        with self._lock:
            start = self._start_state(start_q)
            goal = self._start_state(goal_q)
            res = self.mp.plan_cspace(goal, start, max_attempts=max_attempts)
            return self._result_to_dict(res)

    # ---- world obstacles + carried-object attach ----
    def set_world(self, cuboids):
        """Rebuild the scene = base ground + the given cuboids.
        cuboids: list of {name, dims:[dx,dy,dz], pose:[x,y,z,qw,qx,qy,qz]}."""
        with self._lock:
            cub = dict(self._base_cuboids)
            for c in cuboids:
                cub[c["name"]] = {"dims": list(c["dims"]), "pose": list(c["pose"])}
            self.mp.update_world(SceneCfg.create({"cuboid": cub}))
            return {"success": True, "names": list(cub.keys())}

    def clear_world(self):
        """Reset the scene back to base ground only."""
        with self._lock:
            self.mp.update_world(SceneCfg.create({"cuboid": dict(self._base_cuboids)}))
            return {"success": True, "names": list(self._base_cuboids.keys())}

    def _attachment_managers(self):
        """All per-solver attachment managers (ik + trajopt have separate
        kinematics, so the held object must be attached on each)."""
        mgrs = []
        for solver in (getattr(self.mp, "ik_solver", None),
                       getattr(self.mp, "trajopt_solver", None),
                       getattr(self.mp, "graph_planner", None)):
            core = getattr(solver, "core", None)
            am = getattr(core, "attachment_manager", None)
            if am is not None:
                mgrs.append(am)
        return mgrs

    def attach(self, grasp_q, dims, obj_pose, num_spheres=16):
        """Attach a box (dims, m) at world pose obj_pose (xyz + wxyz, base frame)
        to the tcp using grasp_q for the link FK. The held object's spheres then
        participate in robot collision, so subsequent plans route it around world
        obstacles (e.g. the place box)."""
        with self._lock:
            js = self._start_state(grasp_q)
            cub = Cuboid(name="held_object", pose=[0, 0, 0, 1, 0, 0, 0],
                         dims=list(dims))
            wp = Pose(
                position=torch.tensor([obj_pose[0:3]], dtype=torch.float32,
                                      device=self.device),
                quaternion=torch.tensor([obj_pose[3:7]], dtype=torch.float32,
                                        device=self.device))
            mgrs = self._attachment_managers()
            for am in mgrs:
                am.attach(js, [cub], link_name="attached_object",
                          num_spheres=int(num_spheres),
                          world_objects_pose_offset=wp)
            return {"success": True, "n_spheres": int(num_spheres),
                    "n_managers": len(mgrs)}

    def detach(self):
        """Reset the attached_object link spheres (object no longer carried)."""
        with self._lock:
            for am in self._attachment_managers():
                am.detach(link_name="attached_object")
            return {"success": True}

    # ---- dispatch ----
    def handle(self, req):
        kind = req.get("type")
        if kind == "ping":
            return {"ok": True, "dof": self.dof, "joint_names": self.joint_names,
                    "tool_frame": self.tool_frame, "backend": "curobo_v2"}
        if kind == "plan_pose":
            return self.plan_pose(req["start_q"], req["goal_pose"],
                                  int(req.get("max_attempts", 5)))
        if kind == "plan_joint":
            return self.plan_joint(req["start_q"], req["goal_q"],
                                   int(req.get("max_attempts", 5)))
        if kind == "fk":
            return self.fk(req["q"])
        if kind == "set_world":
            return self.set_world(req.get("cuboids", []))
        if kind == "clear_world":
            return self.clear_world()
        if kind == "attach":
            return self.attach(req["grasp_q"], req["dims"], req["obj_pose"],
                               int(req.get("num_spheres", 16)))
        if kind == "detach":
            return self.detach()
        return {"success": False, "status": f"unknown request type: {kind}"}

    def fk(self, q):
        """Forward kinematics for one or many configs. `q` is [6] or [N,6] rad.
        Returns tcp (tool_frame) position[xyz] + quaternion[wxyz] in base_link."""
        arr = torch.tensor(q, dtype=torch.float32, device=self.device)
        if arr.ndim == 1:
            arr = arr[None, :]
        st = self.mp.compute_kinematics(
            JointState.from_position(arr, joint_names=self.joint_names))
        p = st.tool_poses.get_link_pose(self.tool_frame)
        pos = p.position.detach().cpu().numpy().tolist()
        quat = p.quaternion.detach().cpu().numpy().tolist()  # wxyz
        return {"success": True, "pos": pos, "quat": quat}


def _serve_conn(planner, conn):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    buf = ""
    with conn:
        while True:
            try:
                data = conn.recv(65536)
            except OSError:
                break
            if not data:
                break
            buf += data.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    resp = planner.handle(json.loads(line))
                except Exception as e:
                    resp = {"success": False, "status": f"error: {e}",
                            "traceback": traceback.format_exc()}
                try:
                    conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
                except OSError:
                    return


def serve(planner, host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(4)
    print(f"[planner-v2] listening on {host}:{port}", flush=True)
    try:
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=_serve_conn, args=(planner, conn), daemon=True).start()
    except KeyboardInterrupt:
        print("\n[planner-v2] shutting down", flush=True)
    finally:
        srv.close()


def main():
    p = argparse.ArgumentParser(description="cuRobo V2 motion-planning server for the myCobot Pro 630.")
    p.add_argument("--world", default=None, help="cuRobo V2 scene yaml (obstacles); default = ground plane")
    p.add_argument("--ground-z", type=float, default=-0.1,
                   help="table/ground top height in base_link frame (base spheres reach ~-0.063)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9997)
    p.add_argument("--trajopt-seeds", type=int, default=4)
    p.add_argument("--ik-seeds", type=int, default=32)
    p.add_argument("--collision-activation-distance", type=float, default=0.01)
    args = p.parse_args()

    planner = Planner(ground_z=args.ground_z, world_path=args.world,
                      collision_activation_distance=args.collision_activation_distance,
                      num_trajopt_seeds=args.trajopt_seeds, num_ik_seeds=args.ik_seeds)
    serve(planner, args.host, args.port)


if __name__ == "__main__":
    main()
