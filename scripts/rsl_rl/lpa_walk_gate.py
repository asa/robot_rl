"""Hard walk gate: WORLD displacement under command (tinh-95oj).

Ref-frame tracking metrics are anchored to the robot's own stance
frame and score a standing policy deceptively well (the first LPA
policy 'tracked' at cm level while travelling 0.35 m in 12 s). This
gate demands actual travel: dx >= min_ratio * cmd_vx * duration.

    python -u scripts/rsl_rl/lpa_walk_gate.py \
        "logs/g1_policies/lpa_walking_clf/lpa_walking_clf/*<run>" 0.47
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("run_glob")
parser.add_argument("cmd_vx", nargs="?", type=float, default=0.47)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--min-ratio", type=float, default=0.8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
import yaml, os, glob
import robot_rl.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

env_cfg = parse_env_cfg("LPA-walking-clf-play", device="cuda:0", num_envs=1)
# Pin the command to the gated speed — the play cfg samples
# lin_vel_x in (0.38, 0.57), so without this the bar is compared
# against a speed the policy was never asked for.
env_cfg.commands.base_velocity.ranges.lin_vel_x = (args.cmd_vx, args.cmd_vx)
env = gym.make("LPA-walking-clf-play", cfg=env_cfg)
env = RslRlVecEnvWrapper(env)
run = sorted(glob.glob(args.run_glob))[-1]
ckpts = sorted(glob.glob(os.path.join(run, "model_*.pt")),
               key=lambda p: int(p.split("_")[-1].split(".")[0]))
agent_cfg = yaml.safe_load(open(os.path.join(run, "params/agent.yaml")))
runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device="cuda:0")
# Weights-only: surgery checkpoints (fold/pad) drop the
# optimizer state; gates never optimize.
runner.load(ckpts[-1], load_optimizer=False)
policy = runner.get_inference_policy(device="cuda:0")
print(f"gate on {ckpts[-1]}")

ret = env.get_observations()
obs = ret[0] if isinstance(ret, tuple) else ret
robot = env.unwrapped.scene.articulations["robot"]
x0 = float(robot.data.root_pos_w[0, 0])
for step in range(args.steps):
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
dur = args.steps * 0.02
dx = float(robot.data.root_pos_w[0, 0]) - x0
want = args.min_ratio * args.cmd_vx * dur
print(f"WORLD-DX {dx:+.3f} m over {dur:.0f}s (need >= {want:.2f} at cmd {args.cmd_vx})")
print("WALK-GATE", "PASS" if dx >= want else "FAIL")
app.close()
