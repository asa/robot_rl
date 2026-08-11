"""Turn-in-place gate (tinh-lpa-clfrl.7.9): commanded yaw rate must
produce matching heading change with bounded translation drift.

    python scripts/rsl_rl/lpa_yaw_gate.py <run_glob> [wz] --steps 350
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run_glob")
parser.add_argument("cmd_wz", nargs="?", type=float, default=0.289)
parser.add_argument("--steps", type=int, default=350)
parser.add_argument("--min-ratio", type=float, default=0.7)
parser.add_argument("--max-drift", type=float, default=0.6)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import glob
import os

import gymnasium as gym
import torch
import yaml
import robot_rl.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

env_cfg = parse_env_cfg("LPA-walking-clf-play", device="cuda:0", num_envs=1)
env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.ang_vel_z = (args.cmd_wz, args.cmd_wz)
env = gym.make("LPA-walking-clf-play", cfg=env_cfg)
env = RslRlVecEnvWrapper(env)
run = sorted(glob.glob(args.run_glob))[-1]
ckpts = sorted(glob.glob(os.path.join(run, "model_*.pt")),
               key=lambda p: int(p.split("_")[-1].split(".")[0]))
agent_cfg = yaml.safe_load(open(os.path.join(run, "params/agent.yaml")))
runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device="cuda:0")
runner.load(ckpts[-1])
policy = runner.get_inference_policy(device="cuda:0")
print(f"yaw gate on {ckpts[-1]}")

ret = env.get_observations()
obs = ret[0] if isinstance(ret, tuple) else ret
robot = env.unwrapped.scene.articulations["robot"]
p0 = robot.data.root_pos_w[0, :2].clone()


def _yaw(q):
    w, x, y, z = q
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


yaw_prev = _yaw(robot.data.root_quat_w[0])
yaw_acc = 0.0
import math
for step in range(args.steps):
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    yw = _yaw(robot.data.root_quat_w[0])
    d = float(yw - yaw_prev)
    yaw_acc += (d + math.pi) % (2 * math.pi) - math.pi
    yaw_prev = yw
dur = args.steps * 0.02
drift = float(torch.norm(robot.data.root_pos_w[0, :2] - p0))
want = args.min_ratio * abs(args.cmd_wz) * dur
ok = abs(yaw_acc) >= want and drift <= args.max_drift
print(f"YAW-GATE dyaw {math.degrees(yaw_acc):+.1f} deg over {dur:.0f}s "
      f"(need >= {math.degrees(want if args.cmd_wz > 0 else -want):+.1f}), "
      f"drift {drift:.2f} m (max {args.max_drift})")
print("YAW-GATE", "PASS" if ok else "FAIL")
app.close()
