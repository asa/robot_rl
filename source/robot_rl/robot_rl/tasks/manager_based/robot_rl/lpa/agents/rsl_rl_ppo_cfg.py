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

    # ~16 min between checkpoints at rough-v2 throughput (100 iters x
    # 9.6 s). The 200-iter default meant a mid-run stop lost up to 32
    # minutes; checkpoints are 6 MB, so density is nearly free. Set
    # here rather than the shared G1 base so G1 envs keep upstream
    # behaviour.
    save_interval = 100


@configclass
class PPORunnerCfgH48LowEnt(PPORunnerCfgH48):
    """H48 with the exploration bonus cut 8x: 0.008 -> 0.001.

    WHY (2026-08-28). Training and playback were running different
    robots. Training samples actions from the policy's distribution;
    playback (and deployment) uses the distribution's MEAN. That is
    normal — but this policy widened its own noise std from the 1.0
    it starts at to 2.56, and at that width the two diverge in kind,
    not just degree: the mean action folds the elbow onto its -2.2
    mechanical stop, while the sampled poses training is rewarded on
    average out near -0.9, because nudges past a hard stop do
    nothing and nudges away from it move the arm.

    Consequence: the deep fold in every play video is a state the
    training distribution essentially never visits, so no reward
    defined on that distribution can reach it. Two elbow bands
    (rough029 at -1.20, rough030 at -0.95) both read ~1e-4 proving
    exactly this.

    So the lever is not another penalty — it is closing the gap.
    Nothing ever pushed back on the widening; the entropy bonus pays
    for it. 0.001 is deliberately aggressive for a 500-iteration
    smoke: the question is whether the std moves AT ALL on this
    timescale, and a small step would be unreadable. Tune back up
    once the direction is known.

    The std itself is a LEARNED parameter restored from the resumed
    checkpoint (2.56), so this run starts wide and must earn its way
    down — that decay is the first gate.
    """

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.algorithm.entropy_coef = 0.001

