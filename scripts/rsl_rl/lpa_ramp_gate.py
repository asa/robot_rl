"""Climb gate: does the robot actually gain altitude on each ramp?
(tinh-lpa-ramp.6)

Per terrain column, measures REALIZED climb against the plane:

    dz_actual  vs  tan(theta) * dx_travelled

A robot that walks on the flat portion, or slides back down, fails
even if it never falls. Also reports fall rate and lateral drift per
column, because "didn't fall" is not the same as "climbed".

  python -u scripts/rsl_rl/lpa_ramp_gate.py <run_dir> [--steps 800]
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--num_envs", type=int, default=96)
parser.add_argument("--min-ratio", type=float, default=0.6,
                    help="fraction of the plane's climb that counts")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import glob
import math
import os

import gymnasium as gym
import torch
import yaml

import robot_rl.tasks  # noqa: F401
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from robot_rl.tasks.manager_based.robot_rl.lpa.lpa_walking_clf_env_cfg import (
    LpaWalkingCLFRampEnvCfg,
)
from robot_rl.tasks.manager_based.robot_rl.lpa.lpa_ramp_terrain import (
    RAMP_COLUMNS,
)

env_cfg = LpaWalkingCLFRampEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env = gym.make("LPA-walking-clf-ramp", cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

ckpts = sorted(glob.glob(os.path.join(args.run, "model_*.pt")),
               key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
agent_cfg = yaml.safe_load(open(os.path.join(args.run, "params/agent.yaml")))
runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device="cuda:0")
runner.load(ckpts[-1], load_optimizer=False)
policy = runner.get_inference_policy(device="cuda:0")
print(f"ramp gate on {ckpts[-1]}")

uenv = env.unwrapped
robot = uenv.scene["robot"]
types = uenv.scene.terrain.terrain_types.long()
N = args.num_envs

ret = env.get_observations()
obs = ret[0] if isinstance(ret, tuple) else ret

# Climb is measured at the FEET, not the base.
#
# Two earlier metrics failed because the BASE moves with posture:
# v1 counted each fall's base drop as travel (flat column read
# -3.5 m); v2 filtered on uprightness but the robot crouches 57% of
# the time and the sag leaked through (flat read -0.58 m). The
# stance foot is ON the terrain, so its height IS ground height —
# immune to crouching, sagging, and falls alike. A flat column must
# now read ~0 by construction; that is the check that this metric
# is honest.
feet_idx = [robot.body_names.index(n) for n in ("L_ANKLE", "R_ANKLE")]

def ground_z():
    # stance foot = the lower one = the ground under the robot
    return robot.data.body_pos_w[:, feet_idx, 2].min(dim=1).values

def base_clearance():
    return robot.data.root_pos_w[:, 2] - ground_z()

prev_g = ground_z().clone()
prev_xy = robot.data.root_pos_w[:, :2].clone()
sum_dz = torch.zeros(N, device="cuda:0")
sum_dr = torch.zeros(N, device="cuda:0")
n_up = torch.zeros(N, device="cuda:0")
falls = torch.zeros(N, device="cuda:0")
prev_fall = torch.zeros(N, dtype=torch.bool, device="cuda:0")
tm = uenv.termination_manager

for step in range(args.steps):
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    if "base_height" in tm.active_terms:
        f = tm.get_term("base_height").bool()
        falls += (f & ~prev_fall).float()
        prev_fall = f
    g = ground_z()
    xy = robot.data.root_pos_w[:, :2]
    fresh = uenv.episode_length_buf <= 1          # reset teleport
    z = torch.where(fresh, torch.zeros_like(g), g - prev_g)
    r = torch.where(fresh, torch.zeros_like(g),
                    (xy - prev_xy).norm(dim=1))
    sum_dz += z
    sum_dr += r
    n_up += (base_clearance() > 0.85).float()
    prev_g = g.clone()
    prev_xy = xy.clone()

dz = sum_dz
dr = sum_dr
print(f"upright-step fraction: {float((n_up / args.steps).median()):.2f}")

print(f"{'col':<6} {'envs':>4} {'dz':>7} {'want':>7} {'ratio':>6} "
      f"{'dist':>6} {'falls':>6}")
ok_cols = 0
for i, (n, _, deg) in enumerate(RAMP_COLUMNS):
    m = types == i
    if not int(m.sum()):
        continue
    mdz = float(dz[m].median())
    mdr = float(dr[m].median())
    want = math.tan(math.radians(deg)) * mdr
    ratio = (mdz / want) if abs(want) > 1e-3 else float("nan")
    f = float(falls[m].sum())
    good = (abs(want) < 1e-3) or (ratio >= args.min_ratio)
    ok_cols += int(good)
    print(f"{n:<6} {int(m.sum()):>4} {mdz:>+7.3f} {want:>+7.3f} "
          f"{ratio:>6.2f} {mdr:>6.2f} {f:>6.0f}  "
          f"{'ok' if good else 'LOW'}")

print(f"RAMP-CLIMB columns_ok={ok_cols}/{len(RAMP_COLUMNS)}"
      f" total_falls={int(falls.sum())}")
print("RAMP-GATE", "PASS" if ok_cols == len(RAMP_COLUMNS) else "FAIL")
app.close()
