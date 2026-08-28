"""Dump graphturn18 obs/actions from the live Isaac env — sim2sim
ground truth for the mujoco lpa_drive port (am-5gh.4.5)."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--vx", type=float, default=0.0)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import glob
import os

import gymnasium as gym
import numpy as np
import torch
import yaml

import robot_rl.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

env_cfg = parse_env_cfg("LPA-walking-clf-graphturn", device="cuda:0",
                        num_envs=args.num_envs)
r = env_cfg.commands.base_velocity.ranges
r.lin_vel_x = (args.vx, args.vx)
r.lin_vel_y = (0.0, 0.0)
r.ang_vel_z = (0.0, 0.0)
if hasattr(r, "heading"):
    r.heading = (0.0, 0.0)
# pure locomotion: keep the graph sampler from firing
if hasattr(env_cfg.events, "graph_skills"):
    env_cfg.events.graph_skills.params["p_turn"] = 0.0
env = gym.make("LPA-walking-clf-graphturn", cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

run = sorted(glob.glob(args.run))[-1]
ckpt = sorted(glob.glob(os.path.join(run, "model_*.pt")),
              key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]
print("ckpt:", ckpt)
agent_cfg = yaml.safe_load(open(os.path.join(run, "params/agent.yaml")))
runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device="cuda:0")
try:
    runner.load(ckpt)
except KeyError:
    runner.load(ckpt, load_optimizer=False)
policy = runner.get_inference_policy(device="cuda:0")

cmd = env.unwrapped.command_manager.get_term("traj_ref")
robot = env.unwrapped.scene["robot"]
_o = env.get_observations()
obs = _o[0] if isinstance(_o, tuple) else _o
O, A, PHI, HOLD, Z, EPL = [], [], [], [], [], []
with torch.inference_mode():
    for i in range(args.steps):
        a = policy(obs)
        O.append(obs.cpu().numpy().copy())
        A.append(a.cpu().numpy().copy())
        PHI.append(cmd.get_phasing_var().cpu().numpy().copy())
        HOLD.append(cmd.hold_phi_value.cpu().numpy().copy())
        obs = env.step(a)[0]
        Z.append(robot.data.root_pos_w[:, 2].cpu().numpy().copy())
        EPL.append(env.unwrapped.episode_length_buf.cpu().numpy().copy())
np.savez(args.out, obs=np.array(O), act=np.array(A),
         phi=np.array(PHI), hold=np.array(HOLD), z=np.array(Z), epl=np.array(EPL))
print("saved", args.out, np.array(O).shape)
