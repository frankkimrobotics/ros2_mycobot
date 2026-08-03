#!/usr/bin/env python3
"""curobo_pick_traj :: full press-to-seal pick trajectory from cuRobo phases.

Pipeline (run in the curobo2 env):
  1. Plan each phase with the cuRobo V2 Planner (dynamics-aware trajopt; the
     URDF <inertial> masses/inertias feed its RNEA/torque limits). Every
     phase's B-SPLINE CONTROL POINTS are kept and exported.
  2. STITCH the phase paths geometrically (approach->grasp->press as one
     path, lift->carry->drop as another) and retime each with TOPPRA under
     joint velocity/acceleration limits -- interior phase boundaries are
     passed at NON-ZERO velocity (no stop-and-go between phases).
  3. At contact (press end) insert a DWELL_S stationary hold, and blend the
     profile ends with quintics so position/velocity/acceleration are all
     continuous (C2) into and out of the dwell -- the lift starts smoothly.

Outputs outputs/pick_traj.json: {dt, t, q, qd, qdd, phases, control_points,
continuity report}. Velocity limit defaults to 0.6 rad/s -- the real Pro 630
drive following-error protection faults above ~36 deg/s.

Usage:
    conda activate curobo2
    python curobo_pick_traj.py [--vlim 0.6] [--alim 3.0]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DT = 0.004
DWELL_S = 0.1                     # stationary hold at contact before the lift
BLEND_S = 0.35                    # quintic C2 blend window at profile ends
R_DOWN_QUAT = [0.0, 1.0, 0.0, 0.0]   # wxyz: tcp z-axis pointing down

# demo scene geometry (matches pick_and_place/mjwarp_pick_demo.py)
OBJ_XY = [0.340, 0.050]
OBJ_TOP = 0.060
CUP_R = 0.008
PRESS_M = 0.015
HOVER = 0.12
BIN_XY = [0.10, 0.40]
START_Q = [0.0, -0.349066, 1.396263, 0.174533, -1.570796, 0.0]


def _pose(x, y, z):
    return [x, y, z] + R_DOWN_QUAT


def _yawed(pose, yaw_deg):
    """Rotate a down-pointing goal about the world z (cup is yaw-symmetric)."""
    h = np.radians(yaw_deg) / 2.0
    qy = np.array([np.cos(h), 0.0, 0.0, np.sin(h)])          # wxyz about z
    qd = np.asarray(pose[3:7])
    w = qy[0]*qd[0] - qy[1]*qd[1] - qy[2]*qd[2] - qy[3]*qd[3]
    x = qy[0]*qd[1] + qy[1]*qd[0] + qy[2]*qd[3] - qy[3]*qd[2]
    y = qy[0]*qd[2] - qy[1]*qd[3] + qy[2]*qd[0] + qy[3]*qd[1]
    z = qy[0]*qd[3] + qy[1]*qd[2] - qy[2]*qd[1] + qy[3]*qd[0]
    return pose[:3] + [w, x, y, z]


def plan_phases(P):
    """cuRobo per-phase plans; returns list of (name, traj[N,6], control_points)."""
    hover = _pose(OBJ_XY[0], OBJ_XY[1], OBJ_TOP + HOVER)
    grasp = _pose(OBJ_XY[0], OBJ_XY[1], OBJ_TOP + CUP_R)
    press = _pose(OBJ_XY[0], OBJ_XY[1], OBJ_TOP + CUP_R - PRESS_M)
    bin_h = _pose(BIN_XY[0], BIN_XY[1], 0.42)
    drop = _pose(BIN_XY[0], BIN_XY[1], 0.20)

    phases = []
    q = list(START_Q)
    for name, goal in [("approach", hover), ("descend", grasp), ("press", press),
                       ("lift", hover), ("carry", bin_h), ("drop", drop)]:
        r = None
        for yaw in (0.0, 90.0, -90.0, 180.0, 45.0, -45.0):
            r = P.plan_pose(q, _yawed(goal, yaw), max_attempts=5)
            if r["success"]:
                if yaw:
                    print(f"[plan] {name}: solved with yaw {yaw:+.0f} deg")
                break
        if not r["success"]:
            raise RuntimeError(f"cuRobo failed on phase {name}: {r['status']}")
        traj = np.asarray(r["trajectory"], float)
        phases.append((name, traj, r["control_points"]))
        q = list(traj[-1])
        print(f"[plan] {name}: {len(traj)} pts, {len(r['control_points'])} "
              f"bspline control points, motion_time {r['motion_time']:.2f}s")
    return phases


def _stitch(paths):
    """Concatenate phase paths into one waypoint list (dedup joints)."""
    way = [paths[0][0]]
    for p in paths:
        for w in p:
            if np.linalg.norm(w - way[-1]) > 1e-4:
                way.append(w)
    return np.asarray(way)


def _toppra_retime(way, vlim, alim):
    """TOPPRA retiming of a waypoint path; rest-to-rest but interior waypoints
    keep non-zero velocity. Returns (t, q, qd, qdd) sampled at DT."""
    import toppra as ta
    import toppra.constraint as tac
    # thin the dense cuRobo waypoints: a cubic spline through ~160 near-duplicate
    # points wiggles, and toppra only enforces limits at grid points
    step = max(1, len(way) // 60)
    way = np.vstack([way[::step], way[-1:]])
    keep = [0] + [i for i in range(1, len(way))
                  if np.linalg.norm(way[i] - way[i - 1]) > 1e-4]
    way = way[keep]
    # round the corners at phase junctions (stitched paths join C0): high
    # spline curvature there makes sampled accel overshoot toppra's
    # gridpoint-enforced limits
    for _ in range(3):
        way[1:-1] = (way[:-2] + 2.0 * way[1:-1] + way[2:]) / 4.0
    ss = np.zeros(len(way))
    d = np.linalg.norm(np.diff(way, axis=0), axis=1)
    ss[1:] = np.cumsum(d) / max(d.sum(), 1e-9)
    path = ta.SplineInterpolator(ss, way)
    cv = tac.JointVelocityConstraint(np.stack([-vlim, vlim], axis=1))
    ca = tac.JointAccelerationConstraint(np.stack([-alim, alim], axis=1))
    gp = np.linspace(path.path_interval[0], path.path_interval[1], 1001)
    inst = ta.algorithm.TOPPRA([cv, ca], path, gridpoints=gp,
                               parametrizer="ParametrizeConstAccel")
    jt = inst.compute_trajectory(0, 0)
    t = np.arange(0.0, jt.duration, DT)
    return t, jt(t), jt(t, 1), jt(t, 2)


def _quintic(q0, v0, a0, q1, v1, a1, T, ts):
    """Quintic q(t) matching pos/vel/acc at both ends, evaluated at ts."""
    A = np.array([
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 2, 0, 0],
        [T**5, T**4, T**3, T**2, T, 1],
        [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
        [20*T**3, 12*T**2, 6*T, 2, 0, 0]])
    out = np.zeros((len(ts), len(q0)))
    for j in range(len(q0)):
        c = np.linalg.solve(A, [q0[j], v0[j], a0[j], q1[j], v1[j], a1[j]])
        out[:, j] = np.polyval(c, ts)
    return out


def _c2_end_blends(q, qd, qdd):
    """Quintic-blend both ends of a sampled profile to v=0, a=0 exactly.
    Boundary derivatives come from finite differences of the samples (not the
    parametrizer's analytic ones) so the seam is C2 in the sampled sense."""
    del qd, qdd
    qd_fd = np.gradient(q, DT, axis=0)
    qdd_fd = np.gradient(qd_fd, DT, axis=0)
    nb = max(4, int(BLEND_S / DT))
    ts = np.arange(nb) * DT
    T = ts[-1]
    q[:nb] = _quintic(q[0], np.zeros(6), np.zeros(6),
                      q[nb-1], qd_fd[nb-1], qdd_fd[nb-1], T, ts)
    q[-nb:] = _quintic(q[-nb], qd_fd[-nb], qdd_fd[-nb],
                       q[-1], np.zeros(6), np.zeros(6), T, ts)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlim", type=float, default=0.6,
                    help="rad/s (real Pro 630 drives fault above ~0.63)")
    ap.add_argument("--alim", type=float, default=3.0, help="rad/s^2")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "pick_traj.json"))
    args = ap.parse_args()
    vlim = np.full(6, args.vlim)
    alim = np.full(6, args.alim)

    from curobo_planner_server_v2 import Planner
    P = Planner()
    phases = plan_phases(P)
    names = [n for n, _, _ in phases]

    # stitch: pick = approach+descend+press (one continuous motion, non-zero
    # velocity through the hover and grasp poses); place = lift+carry+drop
    pick_way = _stitch([tr for n, tr, _ in phases[:3]])
    place_way = _stitch([tr for n, tr, _ in phases[3:]])
    t1, q1, qd1, qdd1 = _toppra_retime(pick_way, vlim, alim)
    t2, q2, qd2, qdd2 = _toppra_retime(place_way, vlim, alim)
    q1 = _c2_end_blends(q1, qd1, qdd1)
    q2 = _c2_end_blends(q2, qd2, qdd2)

    # dwell at contact: DWELL_S stationary; both neighbours end/start with
    # v=0, a=0 (quintic blends), so the composite is C2 through the dwell
    nd = int(round(DWELL_S / DT))
    dwell = np.repeat(q1[-1:], nd, axis=0)
    q = np.vstack([q1, dwell, q2])
    t = np.arange(len(q)) * DT
    qd = np.gradient(q, DT, axis=0)
    qdd = np.gradient(qd, DT, axis=0)

    # continuity report at the two stitch instants (end of press / start of lift)
    j0, j1 = len(q1), len(q1) + nd
    dv = np.abs(np.diff(qd, axis=0)).max(axis=1)
    report = {
        "press_end_vel_rad_s": float(np.abs(qd[j0 - 2]).max()),
        "lift_start_vel_rad_s": float(np.abs(qd[j1 + 1]).max()),
        "max_vel_step_anywhere": float(dv.max()),
        "max_vel_rad_s": float(np.abs(qd).max()),
        "max_acc_rad_s2": float(np.abs(qdd).max()),
        "pick_time_s": float(t1[-1]), "dwell_s": DWELL_S, "place_time_s": float(t2[-1]),
    }
    print("[traj] " + json.dumps(report, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "dt": DT, "t": t.tolist(), "q": q.round(6).tolist(),
            "qd": qd.round(6).tolist(), "qdd": qdd.round(6).tolist(),
            "dwell_range": [j0, j1], "vlim": args.vlim, "alim": args.alim,
            "phases": names,
            "control_points": {n: cp for n, _, cp in phases},
            "continuity": report,
        }, f)
    print(f"[traj] wrote {args.out}  ({len(q)} ticks, {t[-1]:.2f}s total)")


if __name__ == "__main__":
    main()
