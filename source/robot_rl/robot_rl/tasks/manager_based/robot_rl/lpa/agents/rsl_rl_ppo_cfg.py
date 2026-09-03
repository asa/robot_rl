# LPA PPO runner variants.
#
# LPA otherwise reuses the G1 runner cfg verbatim (it is robot
# agnostic); this exists only for settings that must differ per run.

from isaaclab.utils import configclass

from ...g1.agents.rsl_rl_ppo_cfg import PPORunnerCfg

# The ONE action bound for every LPA policy, declared in the runner
# cfg so the rsl_rl wrapper applies it identically in train_policy,
# play_policy, play.py and the exported graph, and every probe reads
# it from params/agent.yaml.
#
# It is a RUNAWAY GUARD, not a task element: healthy actions run
# 1-4 (normalized units; the elbow scale is 0.081 rad/unit), the
# gt13 runaway went to 1e12 through the last-action observation loop,
# and 100 stops that while never binding on a healthy policy. The
# previous script-local value of 10 DID bind (the elbow command sat
# on it 70% of the time), which disconnected the elbow's gradient
# from its outcomes and silenced the elbow band -- see
# LpaPPORunnerCfg. If a policy ever saturates this bound, that is a
# finding about the policy, not a reason to tighten the bound.
LPA_CLIP_ACTIONS = 100.0


@configclass
class LpaPPORunnerCfg(PPORunnerCfg):
    """The G1 runner cfg with the LPA action bound declared.

    WHY (am-9j7.1, 2026-09-03). train_policy.py clamped actions to
    +/-10 at the env boundary for every lpa_ env from 2026-08-12 (a
    NaN guard whose comment assumed healthy actions run 1-4 and the
    clamp never binds). Nothing else clamped: play_policy, play.py,
    the lpa_sim probes and the exported policy all executed the raw
    network output. The clamp BOUND on the elbow: with an action scale
    of 0.081 rad/unit it capped the trained elbow near -0.96 rad, and
    the policy learned to saturate the command (cladwalkgaitprog2:
    elbow actions -13 mean, -25 min, 70% beyond the clamp) because
    past the clamp nothing it did changed anything. Deployed
    unclamped, the same policy folded the elbows to -1.24 mean / -2.1
    extremes -- every "arm punch" video and arm-style number between
    2026-08-12 and 2026-09-03 was the deployed policy, not the trained
    one, and the 2026-08-28 diagnosis (PPORunnerCfgH48LowEnt below,
    "mean action vs sampled action at a hard stop") was this same
    mismatch misread.

    THE RULE: the executed action path is part of the task. Any bound,
    clip or transform applied to actions in training is declared HERE
    so every consumer inherits it; a script-local clamp is a second,
    undeclared task. tools/rlrun preflight refuses an LPA registration
    whose runner cfg leaves clip_actions unset.

    THE VALUE: 100, a bound that never binds. The elbow band
    (elbow_depth), the deviation caps and the clearance terms are the
    things that shape the elbow; under the old clamp they could not
    fire. A checkpoint trained under 10 that resumes here will see
    those terms fire hard on its first rollouts (prog2: elbow_depth
    -2.9/step unclamped) -- that is the reward doing the job the
    clamp had been hiding, and the first such resume is the
    measurement of whether it can.

    Checkpoint-compatible: the runner cfg does not touch the network.
    """
    clip_actions = LPA_CLIP_ACTIONS



@configclass
class PPORunnerCfgH48(LpaPPORunnerCfg):
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

    CORRECTION (2026-09-03, am-9j7.1): the gap this docstring
    explains was NOT mean-vs-sampled. train_policy clamped actions to
    +/-10 and play did not; the elbow command saturated the clamp, so
    training held the elbow near -0.96 while every unclamped rollout
    folded it. Measured on cladwalkgaitprog2 at noise std 0.68: mean
    and sampled rollouts agree to 0.01 rad; clamped vs unclamped
    differ by 0.5 rad. See LpaPPORunnerCfg.
    """

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.algorithm.entropy_coef = 0.001


@configclass
class PPORunnerCfgLowEnt(LpaPPORunnerCfg):
    """The entropy cut ALONE: 0.008 -> 0.001, rollout left at 24.

    H48LowEnt bundles this with num_steps_per_env 24 -> 48. Both may
    be right, but bundling them makes a bad result unattributable and
    halves iterations per hour on top. This is the half that the
    cladwalk measurements point at directly.

    WHY, measured on cladwalk2 at ~1000 iterations. The policy widens
    its own action noise from the 1.0 it starts at to 2.00-2.26, and
    at that width the arms cannot be held to the reference:

        joint         ref range   achieved   ratio
        shoulder yaw      0.402      1.179    2.9x
        shoulder pitch    0.600      1.767    2.9x
        shoulder roll     0.080      0.524    6.5x
        elbow             0.300      2.154    7.2x

    The elbow sits at mean -1.52 with a 2.15 rad range, i.e. swinging
    to roughly -2.5 against a -2.2 mechanical stop -- the exact state
    H48LowEnt's docstring describes: "the mean action folds the elbow
    onto its -2.2 mechanical stop, while the sampled poses training is
    rewarded on average out near -0.9."

    And the widening is not a transient. Over clad1's 7482 iterations
    the noise std went 2.37 -> 2.64 while every tracking term was
    flat or falling (joint_pos -17.4%): the entropy bonus was still
    paying to widen long after nothing else improved.

    save_interval from H48 is deliberately NOT inherited here -- that
    is a checkpoint-density choice, not part of the entropy question.
    """

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.algorithm.entropy_coef = 0.001
