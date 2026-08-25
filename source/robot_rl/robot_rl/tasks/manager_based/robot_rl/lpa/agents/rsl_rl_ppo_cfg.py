# LPA PPO runner variants.
#
# LPA otherwise reuses the G1 runner cfg verbatim (it is robot
# agnostic); this exists only for settings that must differ per run.

from isaaclab.utils import configclass

from ...g1.agents.rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class PPORunnerCfgH48(PPORunnerCfg):
    """Double the on-policy rollout: 24 -> 48 steps.

    gamma*lam = 0.99*0.95 = 0.9405, so credit decays with an 11.3-step
    half-life; at 50 Hz control (dt 0.005 x decimation 4) that is
    0.23 s, and a 24-step rollout spans only 0.48 s, over which the
    advantage decays to 0.9405^24 = 23%.

    The arm fling that drives the faceplant precedes torso contact by
    longer than that, so the terminal signal from the fall largely did
    not reach the action responsible; what remained had to be learned
    indirectly through the value function. 48 steps spans ~0.96 s.

    NOTE this halves iterations per wall-clock hour, so an
    iteration-indexed comparison against a 24-step run is NOT
    like-for-like -- compare on wall clock or on env steps.
    """

    num_steps_per_env = 48
