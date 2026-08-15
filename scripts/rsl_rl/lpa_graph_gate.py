"""Graph-traversal turn gate (tinh-lpa-clfrl.7.7 v2).

Turning is a GRAPH TRAVERSAL (user direction 2026-08-14), not a
vel_yaw command — so the gate runs the graphturn env with its
sampler hot (high p_turn) and scores completed traversals directly:

  * every env completes >= 1 stop -> turn -> start sequence
  * per-sequence base-yaw delta within tol of the reference turn
    (turn cycle vel_yaw 0.269 rad/s x turn_periods x period)
  * no waist_twist / runaway terminations during the run

    python -u scripts/rsl_rl/lpa_graph_gate.py <run_dir> --steps 1500
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--yaw-tol", type=float, default=0.45,
                    help="per-sequence |yaw error| bound (rad)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import glob
import os

import gymnasium as gym
import torch
import yaml

import robot_rl.tasks  # noqa: F401
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from robot_rl.tasks.manager_based.robot_rl.lpa.lpa_walking_clf_env_cfg import (
    LpaWalkingCLFGraphTurnEnvCfg,
)

env_cfg = LpaWalkingCLFGraphTurnEnvCfg()
env_cfg.scene.num_envs = args.num_envs
# Hot sampler: every free-walking env starts a turn sequence quickly.
env_cfg.events.graph_skills.params["p_turn"] = 0.25
env = gym.make("LPA-walking-clf-graphturn", cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

ckpts = sorted(glob.glob(os.path.join(args.run, "model_*.pt")),
               key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
agent_cfg = yaml.safe_load(open(os.path.join(args.run, "params/agent.yaml")))
runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device="cuda:0")
runner.load(ckpts[-1], load_optimizer=False)
policy = runner.get_inference_policy(device="cuda:0")
print(f"graph gate on {ckpts[-1]}")

uenv = env.unwrapped
robot = uenv.scene["robot"]
cmd = uenv.command_manager.get_term("traj_ref")

_TURNING = 2


def base_yaw():
    return euler_xyz_from_quat(robot.data.root_quat_w)[2]


ret = env.get_observations()
obs = ret[0] if isinstance(ret, tuple) else ret
N = args.num_envs
prev_turning = torch.zeros(N, dtype=torch.bool, device="cuda:0")
yaw_at_entry = torch.zeros(N, device="cuda:0")
seq_count = torch.zeros(N, dtype=torch.long, device="cuda:0")
deltas = []
bad_terms = 0
for step in range(args.steps):
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    st = getattr(cmd, "_graph_state", None)
    if st is None:
        # The sampler creates its state lazily on the first
        # interval tick (~0.5 s in) — nothing to score yet.
        continue
    turning = st == _TURNING
    entered = turning & ~prev_turning
    exited = ~turning & prev_turning
    if entered.any():
        yaw_at_entry[entered] = base_yaw()[entered]
    if exited.any():
        d = wrap_to_pi(base_yaw()[exited] - yaw_at_entry[exited])
        for v in d.tolist():
            deltas.append(v)
        seq_count[exited] += 1
    prev_turning = turning
    tm = uenv.termination_manager
    for name in ("waist_twist", "runaway"):
        if name in tm.active_terms:
            bad_terms += int(tm.get_term(name).sum())
    if step % 300 == 299:
        print(f"step {step+1}: sequences {int(seq_count.sum())}"
              f" (envs with >=1: {int((seq_count > 0).sum())}/{N})"
              f" bad-terms {bad_terms}")

# Expected magnitude: turn cycle vel_yaw x periods x cycle time.
turn_periods = env_cfg.events.graph_skills.params.get("turn_periods", 2.0)
expect = 0.269 * turn_periods * 2.4
dt = torch.tensor(deltas) if deltas else torch.zeros(0)
ok_cov = int((seq_count > 0).sum()) == N
err = (dt.abs() - expect).abs() if len(dt) else torch.zeros(0)
ok_yaw = len(dt) > 0 and bool((err <= args.yaw_tol).float().mean() >= 0.8)
ok_term = bad_terms == 0
print(f"GRAPH-GATE sequences={len(dt)} envs-covered="
      f"{int((seq_count > 0).sum())}/{N}"
      f" |dyaw| median={dt.abs().median():.3f} expect={expect:.3f}"
      f" within-tol={float((err <= args.yaw_tol).float().mean() if len(dt) else 0):.2f}"
      f" bad-terms={bad_terms}" if len(dt) else
      f"GRAPH-GATE sequences=0 envs-covered=0/{N} bad-terms={bad_terms}")
print("GRAPH-GATE", "PASS" if (ok_cov and ok_yaw and ok_term) else "FAIL")
app.close()
